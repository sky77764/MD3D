import numpy as np
import torch
import torch.nn as nn
import math

from torch.nn import functional as func
from ...utils import md3d_func, loss_utils
from ...utils import md3d_utils as lib_util
import torch.nn.functional as F
from ...utils import box_coder_utils, box_utils, common_utils, loss_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils


class PointMixtureDensityHead(nn.Module):
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sigmoid_prob = self.model_cfg.SIGMOID_PROB
        self.wo_sampling_mode = (self.model_cfg.LOSS_CONFIG.LOSS_FUNC_ARGS['lw_dict']['mod_nll'] == 0)

        self.num_class = num_class
        self.predict_boxes_when_training = predict_boxes_when_training
        self.infer_pi_thresh = model_cfg.INFER_PI_THRESH
        self.mixture_component_mult = model_cfg.MIXTURE_COMPONENT_MULT

        self.forward_ret_dict = {}
        self.output_sizes = None
        self.num_filters = model_cfg.NUM_FILTERS

        # channels for reg branch
        self.split_indices = [7, 7, 1]
        self.dir_reg_channel = 1
        self.use_dir_sincos = self.model_cfg.USE_SINCOS_DIR
        if self.use_dir_sincos:
            self.dir_reg_channel = 2
            self.split_indices[0] += 1
            self.split_indices[1] += 1

        # encode & decode boxes with two corners and length
        self.use_corner_reg = False
        if self.model_cfg.get('USE_CORNER_REG', False) is not False:
            self.use_corner_reg = True

        self.use_dir_cls = False
        # if not self.use_corner_reg and self.model_cfg.get('USE_DIRECTION_CLASSIFIER', False) is not False:
        #     self.use_dir_cls = True
        #     self.num_dir_bins = self.model_cfg.NUM_DIR_BINS
        #     self.split_indices.append(self.num_dir_bins)

        # Multivariate
        self.multivariate = self.model_cfg.MULTIVARIATE
        self.multivariate_split_ch = self.model_cfg.MULTIVARIATE_SPLIT_CH
        if self.multivariate:
            self.o2_split_ch = [int(n * (n + 1) / 2) for n in self.multivariate_split_ch]
            self.split_indices[1] = sum(self.o2_split_ch)

        self.split_indices = [idx * self.mixture_component_mult for idx in self.split_indices]
        
        self.output_ch = sum(self.split_indices)

        # self.linformer = None
        # if model_cfg.LINFORMER:
        #     self.linformer = Linformer(
        #         input_size=16384,
        #         channels=input_channels,
        #         dim_k=256,
        #         nhead=4,
        #         full_attention=False,
        #         include_ff=False
        #     )

        self.box_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=input_channels,
            output_channels=self.output_ch
        )
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class
        )

        self.loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS

        global_args = {'n_classes': self.num_class}
        loss_func_args = self.model_cfg.LOSS_CONFIG.LOSS_FUNC_ARGS
        loss_func_args['sigmoid_prob'] = self.sigmoid_prob
        loss_func_args['use_sincos_dir'] = self.use_dir_sincos
        loss_func_args['multivariate'] = self.multivariate
        loss_func_args['dir_offset'] = self.model_cfg.DIR_OFFSET
        loss_func_args['dir_limit_offset'] = self.model_cfg.DIR_LIMIT_OFFSET
        loss_func_args['num_dir_bins'] = self.model_cfg.NUM_DIR_BINS
        loss_func_args['use_corner_reg'] = self.use_corner_reg

        for key, value in loss_func_args.items():
            if value == 'None':
                loss_func_args[key] = None
        self.box_loss_func = loss_utils.MD3DPointLossFunc(global_args=global_args, loss_func_args=loss_func_args)
        self.cls_loss_func = loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)


    @staticmethod
    def make_fc_layers(fc_cfg, input_channels, output_channels):
        fc_layers = []
        c_in = input_channels
        for k in range(0, fc_cfg.__len__()):
            fc_layers.extend([
                # nn.Conv1d(c_in, fc_cfg[k], 1, 1, 0, bias=False),
                nn.Linear(c_in, fc_cfg[k], bias=False),
                nn.BatchNorm1d(fc_cfg[k]),
                nn.ReLU(),
            ])
            c_in = fc_cfg[k]
        # fc_layers.append(nn.Conv1d(c_in, output_channels, 1, 1, 0, bias=True))
        fc_layers.append(nn.Linear(c_in, output_channels, bias=True))
        return nn.Sequential(*fc_layers)

    def __decode__(self, o1, point_xyz=None):
        if self.use_corner_reg:
            o1_xyz_flt, o1_xyz_brb, o1_w = torch.split(o1, [3, 3, 1], dim=1)
            if point_xyz is not None:
                o1_xyz_flt = o1_xyz_flt + point_xyz
                o1_xyz_brb = o1_xyz_brb + point_xyz
            o1_w = F.relu(o1_w)
            mu = torch.cat([o1_xyz_flt, o1_xyz_brb, o1_w], dim=1)
        else:
            o1_xy, o1_z, o1_lwh, o1_ry = torch.split(o1, [2, 1, 3, self.dir_reg_channel], dim=1)
            if point_xyz is not None:
                o1_xy = o1_xy + point_xyz[:, :2].detach()
                o1_z = o1_z + point_xyz[:, 2:3].detach()

            if self.use_dir_cls:
                o1_ry = F.relu(o1_ry)
            mu = torch.cat([o1_xy, o1_z, o1_lwh, o1_ry], dim=1)
        return mu

    def __get_mixture_params__(self, point_box_preds, point_xyz=None):
        if self.use_dir_cls:
            out_mu, out_sig, out_pi, dir_cls_pred = torch.split(point_box_preds['out_reg'], self.split_indices, dim=1)
        else:
            out_mu, out_sig, out_pi = torch.split(point_box_preds['out_reg'], self.split_indices, dim=1)

        if self.mixture_component_mult > 1:
            exit()
        #     o1 = o1.reshape([o1.shape[0], o1.shape[1] // self.mixture_component_mult, -1])
        #     o2 = o2.reshape([o2.shape[0], o2.shape[1] // self.mixture_component_mult, -1])
        #     o3 = o3.reshape([o3.shape[0], o3.shape[1] // self.mixture_component_mult, -1])

        mu = self.__decode__(out_mu, point_xyz)

        if self.multivariate:
            o2_split = list(torch.split(out_sig, self.o2_split_ch, dim=1))
            U, sqrt_inv_det_cov = [], []
            for i in range(len(self.multivariate_split_ch)):
                triu_indices = torch.triu_indices(self.multivariate_split_ch[i], self.multivariate_split_ch[i]).cuda()
                diag_indices = torch.arange(0, self.multivariate_split_ch[i], dtype=torch.int64).cuda()
                U.append(torch.zeros([o2_split[i].shape[0], self.multivariate_split_ch[i], self.multivariate_split_ch[i]]).cuda())
                U[i][:, triu_indices[0], triu_indices[1]] = o2_split[i]
                U[i][:, diag_indices, diag_indices] = torch.relu(U[i][:, diag_indices, diag_indices]) + lib_util.epsilon
                sqrt_inv_det_cov.append(torch.prod(U[i][:, diag_indices, diag_indices], dim=1, keepdim=True))
            sig = None

        else:
            sig_min = torch.ones_like(mu) * lib_util.epsilon
            sig = torch.max(func.softplus(out_sig), sig_min)

        pi = func.softmax(out_pi, dim=0)

        param_dict = {'mu': mu, 'sig': sig, 'pi': pi}

        if self.multivariate:
            param_dict.update({'sqrt_inv_det_cov': sqrt_inv_det_cov, 'U': U})

        if self.use_dir_cls:
            param_dict.update({'dir_cls_preds': func.softmax(dir_cls_pred, dim=1)})

        return param_dict

    def get_cls_layer_loss(self, tb_dict=None):
        point_cls_labels = self.forward_ret_dict['data_dict']['point_cls_labels'].view(-1)
        point_cls_preds = self.forward_ret_dict['data_dict']['batch_cls_preds'].view(-1, self.num_class)

        positives = (point_cls_labels > 0)
        negative_cls_weights = (point_cls_labels == 0) * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = point_cls_preds.new_zeros(*list(point_cls_labels.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (point_cls_labels * (point_cls_labels >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(point_cls_preds, one_hot_targets, weights=cls_weights)
        point_loss_cls = cls_loss_src.sum()

        point_loss_cls = point_loss_cls * self.loss_weights_dict['point_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'point_loss_cls': point_loss_cls.item()#,
            # 'point_pos_num': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict

    def get_loss(self):
        loss_dict, value_dict = self.box_loss_func.forward(self.forward_ret_dict)
        mog_nll, point_loss_cls = 0, 0
        tb_dict = {}
        if 'mog_nll' in loss_dict.keys():
            mog_nll = loss_dict["mog_nll"]
            mog_nll = torch.sum(mog_nll, dim=0) / mog_nll.shape[0]
            mog_nll = mog_nll * self.loss_weights_dict['point_box_weight']
            tb_dict['mog_nll'] = mog_nll.item()

        point_loss_cls, tb_dict_cls = self.get_cls_layer_loss()
        tb_dict.update(tb_dict_cls)

        rpn_loss = mog_nll + point_loss_cls
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, output_dict, pi_thresholding=True):
        mu = output_dict['mu']
        pi = output_dict['pi']
        dir_cls_preds = None
        if 'dir_cls_preds' in output_dict.keys():
            dir_cls_preds = output_dict['dir_cls_preds']

        batch_box_preds = mu#.transpose(1, 2)
        if not self.use_corner_reg:
            if self.use_dir_sincos:
                dir = torch.atan2(batch_box_preds[..., 6], batch_box_preds[..., 7]).unsqueeze(-1)
                batch_box_preds = torch.cat((batch_box_preds[..., :6], dir), dim=-1)

            if dir_cls_preds is not None:
                dir_offset = self.model_cfg.DIR_OFFSET
                dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
                # dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1)
                dir_labels = torch.max(dir_cls_preds, dim=1)[1]

                period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
                dir_rot = common_utils.limit_period(
                    batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
                )
                batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)
        else:
            batch_box_preds = box_utils.decode_fltbrbw_to_boxes(batch_box_preds)

        fg_mask = torch.ones_like(pi.squeeze(dim=1), dtype=torch.bool)
        norm_pi = (pi / torch.max(pi, dim=0, keepdim=True)[0]).squeeze(dim=1)
        if pi_thresholding and self.infer_pi_thresh > 0:
            pi_keep_mask = norm_pi > self.infer_pi_thresh
            fg_mask = fg_mask & pi_keep_mask

        # batch_index = []
        # for i in range(fg_mask.shape[0]):
        #     num_fg = batch_cls_preds[i][fg_mask[i]].shape[0]
        #     batch_index.append(torch.ones(num_fg, device=fg_mask.device) * i)
        # batch_index = torch.cat(batch_index)
        #
        # if not self.wo_sampling_mode:
        #     if self.sigmoid_prob:
        #         batch_cls_preds = batch_cls_preds[fg_mask][:]
        #     else:
        #         batch_cls_preds = batch_cls_preds[fg_mask][:, 1:]
        # else:
        #     batch_cls_preds = batch_cls_preds * norm_pi.unsqueeze(-1)
        #     batch_cls_preds = batch_cls_preds[fg_mask]
        # batch_box_preds = batch_box_preds[fg_mask]

        return batch_box_preds, fg_mask


    def assign_targets(self, input_dict):
        """
        Args:
            input_dict:
                point_features: (N1 + N2 + N3 + ..., C)
                batch_size:
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                gt_boxes (optional): (B, M, 8)
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_part_labels: (N1 + N2 + N3 + ..., 3)
        """
        point_coords = input_dict['point_coords']
        gt_boxes = input_dict['gt_boxes']
        assert gt_boxes.shape.__len__() == 3, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert point_coords.shape.__len__() in [2], 'points.shape=%s' % str(point_coords.shape)

        batch_size = gt_boxes.shape[0]
        extend_gt_boxes = box_utils.enlarge_box3d(
            gt_boxes.view(-1, gt_boxes.shape[-1]), extra_width=[0.2, 0.2, 0.2]
        ).view(batch_size, -1, gt_boxes.shape[-1])
        targets_dict = self.assign_stack_targets(
            points=point_coords, gt_boxes=gt_boxes, extend_gt_boxes=extend_gt_boxes,
            set_ignore_flag=True, use_ball_constraint=False,
            ret_part_labels=False, ret_box_labels=False
        )

        return targets_dict

    def assign_stack_targets(self, points, gt_boxes, extend_gt_boxes=None,
                             ret_box_labels=False, ret_part_labels=False,
                             set_ignore_flag=True, use_ball_constraint=False, central_radius=2.0):
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8]
            ret_box_labels:
            ret_part_labels:
            set_ignore_flag:
            use_ball_constraint:
            central_radius:

        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        assert len(points.shape) == 2 and points.shape[1] == 4, 'points.shape=%s' % str(points.shape)
        assert len(gt_boxes.shape) == 3 and gt_boxes.shape[2] == 8, 'gt_boxes.shape=%s' % str(gt_boxes.shape)
        assert extend_gt_boxes is None or len(extend_gt_boxes.shape) == 3 and extend_gt_boxes.shape[2] == 8, \
            'extend_gt_boxes.shape=%s' % str(extend_gt_boxes.shape)
        assert set_ignore_flag != use_ball_constraint, 'Choose one only!'
        batch_size = gt_boxes.shape[0]
        bs_idx = points[:, 0]
        point_cls_labels = points.new_zeros(points.shape[0]).long()
        point_box_labels = gt_boxes.new_zeros((points.shape[0], 8)) if ret_box_labels else None
        point_part_labels = gt_boxes.new_zeros((points.shape[0], 3)) if ret_part_labels else None
        for k in range(batch_size):
            bs_mask = (bs_idx == k)
            points_single = points[bs_mask][:, 1:4]
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
            box_fg_flag = (box_idxs_of_pts >= 0)
            if set_ignore_flag:
                extend_box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    points_single.unsqueeze(dim=0), extend_gt_boxes[k:k+1, :, 0:7].contiguous()
                ).long().squeeze(dim=0)
                fg_flag = box_fg_flag
                ignore_flag = fg_flag ^ (extend_box_idxs_of_pts >= 0)
                point_cls_labels_single[ignore_flag] = -1
            elif use_ball_constraint:
                box_centers = gt_boxes[k][box_idxs_of_pts][:, 0:3].clone()
                box_centers[:, 2] += gt_boxes[k][box_idxs_of_pts][:, 5] / 2
                ball_flag = ((box_centers - points_single).norm(dim=1) < central_radius)
                fg_flag = box_fg_flag & ball_flag
            else:
                raise NotImplementedError

            gt_box_of_fg_points = gt_boxes[k][box_idxs_of_pts[fg_flag]]
            point_cls_labels_single[fg_flag] = 1 if self.num_class == 1 else gt_box_of_fg_points[:, -1].long()
            point_cls_labels[bs_mask] = point_cls_labels_single

            if ret_box_labels and gt_box_of_fg_points.shape[0] > 0:
                point_box_labels_single = point_box_labels.new_zeros((bs_mask.sum(), 8))
                fg_point_box_labels = self.box_coder.encode_torch(
                    gt_boxes=gt_box_of_fg_points[:, :-1], points=points_single[fg_flag],
                    gt_classes=gt_box_of_fg_points[:, -1].long()
                )
                point_box_labels_single[fg_flag] = fg_point_box_labels
                point_box_labels[bs_mask] = point_box_labels_single

            if ret_part_labels:
                point_part_labels_single = point_part_labels.new_zeros((bs_mask.sum(), 3))
                transformed_points = points_single[fg_flag] - gt_box_of_fg_points[:, 0:3]
                transformed_points = common_utils.rotate_points_along_z(
                    transformed_points.view(-1, 1, 3), -gt_box_of_fg_points[:, 6]
                ).view(-1, 3)
                offset = torch.tensor([0.5, 0.5, 0.5]).view(1, 3).type_as(transformed_points)
                point_part_labels_single[fg_flag] = (transformed_points / gt_box_of_fg_points[:, 3:6]) + offset
                point_part_labels[bs_mask] = point_part_labels_single

        targets_dict = {
            'point_cls_labels': point_cls_labels,
            'point_box_labels': point_box_labels,
            'point_part_labels': point_part_labels
        }
        return targets_dict

    def forward(self, batch_dict):
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']

        point_cls_preds = self.cls_layers(point_features)
        out_tensors = {'out_reg': self.box_layers(point_features)}

        output_dict = self.__get_mixture_params__(out_tensors, point_xyz=batch_dict['point_coords'][..., 1:])
        batch_dict.update(output_dict)
        point_box_preds, pi_fg_mask = self.generate_predicted_boxes(batch_dict)

        point_cls_preds_max, _ = point_cls_preds.max(dim=1)
        batch_dict['point_cls_scores'] = torch.sigmoid(point_cls_preds_max)

        batch_dict['batch_box_preds'] = point_box_preds.reshape([batch_dict['batch_size'], -1, point_box_preds.shape[-1]])
        batch_dict['batch_cls_preds'] = point_cls_preds.reshape([batch_dict['batch_size'], -1, point_cls_preds.shape[-1]])
        batch_dict['pi_fg_mask'] = pi_fg_mask.reshape([batch_dict['batch_size'], -1])

        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['point_cls_labels'] = targets_dict['point_cls_labels']
            self.forward_ret_dict['data_dict'] = batch_dict
            self.forward_ret_dict['output_dict'] = output_dict
            return batch_dict

        if not self.training or self.predict_boxes_when_training:
            # batch_cls_preds, batch_box_preds, batch_index, batch_pi_norm = self.generate_predicted_boxes(output_dict)
            # batch_cls_preds, batch_box_preds, batch_index, batch_pi_norm, batch_dict['sample_boxes'] = self.generate_predicted_boxes(output_dict)
            # batch_dict['batch_cls_preds'] = batch_cls_preds
            # batch_dict['batch_box_preds'] = batch_box_preds
            # batch_dict['batch_index'] = batch_index
            # batch_dict['cls_preds_normalized'] = True
            # batch_dict['batch_pi_norm'] = batch_pi_norm
            return batch_dict

