import numpy as np
import torch
import torch.nn as nn
import math

from torch.nn import functional as func
from ...utils import md3d_func, loss_utils, common_utils, box_utils
from ...utils import md3d_utils as lib_util
import torch.nn.functional as F
from pcdet.models.model_utils import centernet_utils


class BEVMixtureDensityHead(nn.Module):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sigmoid_prob = self.model_cfg.SIGMOID_PROB if 'SIGMOID_PROB' in self.model_cfg else False
        self.remove_cls_branch = self.model_cfg.REMOVE_CLS_BRANCH if 'REMOVE_CLS_BRANCH' in self.model_cfg else False
        self.separate_pi_per_class = model_cfg.SEPARATE_PI_PER_CLASS if 'SEPARATE_PI_PER_CLASS' in self.model_cfg else False
        self.fuse_pi_w_prob = self.model_cfg.FUSE_PI_W_PROB if 'FUSE_PI_W_PROB' in self.model_cfg else False
        self.fuse_alpha = self.model_cfg.FUSE_ALPHA if 'FUSE_ALPHA' in self.model_cfg else 0.0

        if self.remove_cls_branch:
            self.sigmoid_prob = True
        if not self.sigmoid_prob:
            self.num_class = num_class + 1
        else:
            self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.infer_pi_thresh = model_cfg.INFER_PI_THRESH
        self.infer_remove_bg = model_cfg.INFER_REMOVE_BG
        self.infer_get_local_peak = model_cfg.INFER_GET_LOCAL_PEAK
        self.infer_pi_fuse_alpha = model_cfg.INFER_PI_FUSE_ALPHA if 'INFER_PI_FUSE_ALPHA' in self.model_cfg else 0.0
        self.infer_pi_sig_fuse_alpha = model_cfg.INFER_PI_SIG_FUSE_ALPHA if 'INFER_PI_SIG_FUSE_ALPHA' in self.model_cfg else 0.0
        self.mixture_component_mult = model_cfg.MIXTURE_COMPONENT_MULT
        self.separate_classes_ch = True #model_cfg.SEPARATE_CLASSES_CH if self.mixture_component_mult > 1 else False
        self.shared_pi_prob = model_cfg.SHARED_PI_PROB if 'SHARED_PI_PROB' in self.model_cfg else False

        self.forward_ret_dict = {}

        self.point_cloud_range = point_cloud_range
        self.coord_scale_factor = self.model_cfg.XY_COORD_SCALE_FACTOR
        self.input_size = np.flip(grid_size[:2])    # yx
        self.coord_range = [    # yx
            self.input_size[0] / self.coord_scale_factor,
            self.input_size[1] / self.coord_scale_factor,
        ]
        self.voxel_size = [
            (point_cloud_range[3] - point_cloud_range[0]) / self.input_size[1],
            (point_cloud_range[4] - point_cloud_range[1]) / self.input_size[0]
        ]

        self.xy_lidar_to_bev = self.coord_range / np.flip((point_cloud_range[3:5] - point_cloud_range[:2])) # yx

        self.forward_ret_dict["point_cloud_range"] = self.point_cloud_range
        self.forward_ret_dict["coord_range"] = self.coord_range
        self.forward_ret_dict['xy_lidar_to_bev'] = self.xy_lidar_to_bev

        self.output_sizes = None
        self.num_fmap_ch = input_channels

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

        # encode & decode boxes with center, front center, width and length
        self.use_front_center_reg = False
        if self.model_cfg.get('USE_FRONT_CENTER_REG', False) is not False and not self.use_corner_reg:
            self.use_front_center_reg = True

        self.use_dir_cls = False

        # Multivariate
        self.multivariate = self.model_cfg.MULTIVARIATE
        self.multivariate_split_ch = self.model_cfg.MULTIVARIATE_SPLIT_CH
        if self.multivariate:
            self.o2_split_ch = [int(n * (n + 1) / 2) for n in self.multivariate_split_ch]
            self.split_indices[1] = sum(self.o2_split_ch)

        self.split_indices = [idx * self.mixture_component_mult for idx in self.split_indices]
        if self.mixture_component_mult > 1 and self.shared_pi_prob:
            self.split_indices[-1] = 1

##############
        # self.output_reg_ch = sum(self.split_indices)
        self.output_reg_ch = self.split_indices[0] + self.split_indices[2]
        self.output_reg_sig_ch = self.split_indices[1]
#############
        self.feature_scales = self.model_cfg.FEATURE_SCALES if "FEATURE_SCALES" in self.model_cfg else [0.5]

        self.reg_branch = nn.Conv2d(self.num_fmap_ch, self.output_reg_ch, 1, 1, 0, bias=True)
##############
        if self.training:
            self.reg_sig_branch = nn.Conv2d(self.num_fmap_ch, self.output_reg_sig_ch, 1, 1, 0, bias=True)
##############
        self.output_cls_ch = self.num_class
        if not self.remove_cls_branch:
            if not self.shared_pi_prob:
                self.output_cls_ch *= self.mixture_component_mult
            self.cls_branch = nn.Conv2d(self.num_fmap_ch, self.output_cls_ch, 1, 1, 0, bias=True)
            self.cls_branch.bias.data.fill_(-2.19)

        output_sizes = list()
        for r in self.feature_scales:
            f_map_y = math.ceil(self.input_size[0] * r)
            f_map_x = math.ceil(self.input_size[1] * r)
            output_sizes.append((f_map_y, f_map_x))
        self.xy_lidar_to_output = np.array(output_sizes[0]) / np.flip(point_cloud_range[3:5] - point_cloud_range[:2])
        self.forward_ret_dict['xy_lidar_to_output'] = self.xy_lidar_to_output

        self.lidar_coord_reg = self.model_cfg.LIDAR_COORD_REG if "LIDAR_COORD_REG" in self.model_cfg else True
        if self.use_corner_reg or self.use_front_center_reg:
            self.lidar_coord_reg = True
        self.center_offset = list()
        for i, _ in enumerate(self.feature_scales):
            if self.lidar_coord_reg:
                center_offset_i = torch.from_numpy(
                    lib_util.create_coord_map_lidar(output_sizes[i], self.coord_range, self.xy_lidar_to_bev,
                                                    self.point_cloud_range))
            else:
                center_offset_i = torch.from_numpy(lib_util.create_coord_map(output_sizes[i], self.coord_range))

            center_offset_i = center_offset_i.view(1, 2, -1)
            self.center_offset.append(center_offset_i)

        self.center_offset = torch.cat(self.center_offset, dim=2).cuda().detach()

        global_args = {'n_classes': self.num_class}
        loss_func_args = self.model_cfg.LOSS_CONFIG.LOSS_FUNC_ARGS
        loss_func_args['use_dir_sincos'] = self.use_dir_sincos
        loss_func_args['sigmoid_prob'] = self.sigmoid_prob
        loss_func_args['multivariate'] = self.multivariate
        loss_func_args['separate_classes_ch'] = self.separate_classes_ch
        loss_func_args['remove_cls_branch'] = self.remove_cls_branch
        loss_func_args['lidar_coord_reg'] = self.lidar_coord_reg
        loss_func_args['use_corner_reg'] = self.use_corner_reg
        loss_func_args['use_front_center_reg'] = self.use_front_center_reg

        for key, value in loss_func_args.items():
            if value == 'None':
                loss_func_args[key] = None

        # self.centerpoint_cls = False
        # self.hm_loss_func = None
        # if 'centerpoint_cls' in loss_func_args.keys() and loss_func_args['centerpoint_cls']:
        #     self.centerpoint_cls = loss_func_args['centerpoint_cls']
        #     self.hm_loss_func = loss_utils.FocalLossCenterNet()

        # auxiliary corner loss
        self.auxiliary_corner_loss = False
        # if 'aux_corners_mog_nll' in loss_func_args['lw_dict']:
        #     if loss_func_args['lw_dict']['aux_corners_mog_nll'] > 0.0:
        #         self.auxiliary_corner_loss = True
        #         self.auxiliary_share_pi = False
        #         if self.auxiliary_share_pi:
        #             # self.aux_split_indices = [24, 24, 1]
        #             self.aux_split_indices = self.split_indices[:-1]
        #         else:
        #             self.aux_split_indices = self.split_indices
        #         self.aux_output_ch = sum(self.aux_split_indices)
        #         self.aux_branch = nn.Conv2d(self.num_fmap_ch, self.aux_output_ch, 1, 1, 0, bias=True)

        self.loss_func = loss_utils.MD3DLossFunc(global_args=global_args, loss_func_args=loss_func_args)
        self.init_weights(set_reg_bias=model_cfg.get('REG_BIAS', False))

    def init_weights(self, weight_init='xavier', set_reg_bias=False):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                # if m.bias is not None:
                #     nn.init.constant_(m.bias, 0)

        # if set_reg_bias:
        #     default_lwh_bias = [[3.9, 1.6, 1.56], [0.8, 0.6, 1.73], [1.76, 0.6, 1.73]]
        #     mu_interval = int(self.split_indices[0] / self.mixture_component_mult)
        #     # sig_interval = int(self.split_indices[1] / self.mixture_component_mult)
        #     for i in range(self.mixture_component_mult):
        #         for j in range(3):
        #             # self.reg_branch.bias[mu_interval*i:mu_interval*(i+1)][3+j].fill_(default_lwh_bias[i][j])
        #             nn.init.constant_(self.reg_branch.bias[mu_interval*i:mu_interval*(i+1)][3+j], default_lwh_bias[i][j])

    def __get_output_tensors__(self, fmap, batch_size):
        out_tensor_reg = self.reg_branch.forward(fmap)
        out_tensor_reg = out_tensor_reg.view((batch_size, self.output_reg_ch, -1))
        if not self.remove_cls_branch:
            out_tensor_cls = self.cls_branch.forward(fmap)
            out_tensor_cls = out_tensor_cls.view((batch_size, self.output_cls_ch, -1))
        # out_tensors = torch.split(out_tensor_reg, self.split_indices, dim=1)
        out_tensors = torch.split(out_tensor_reg, [self.split_indices[0], self.split_indices[2]], dim=1)
        # out_tensors = list(out_tensors)
        out_tensors = {'out_reg': out_tensors}
#########
        if self.training:
            out_tensor_reg_sig = self.reg_sig_branch.forward(fmap)
            out_tensor_reg_sig = out_tensor_reg_sig.view((batch_size, self.output_reg_sig_ch, -1))
            out_tensors.update({'out_reg_sig': out_tensor_reg_sig})
##########
        if not self.remove_cls_branch:
            # out_tensors.insert(2, out_tensor_cls)
            out_tensors.update({'out_cls': out_tensor_cls})

        # if self.learnable_pi_cap:
        #     pooled_fmap = F.adaptive_avg_pool2d(fmap, 1).squeeze(-1)
        #     pi_cap_per_class = self.learnable_pi_mlp(pooled_fmap)
        #     pi_cap_per_class = F.softmax(pi_cap_per_class, dim=1)
        #     # out_tensors.append(pi_cap_per_class)
        #     out_tensors.update({'out_pi_cap_per_class': pi_cap_per_class})

        # if self.use_dir_cls:
        #     dir_cls_preds = self.conv_dir_cls(fmap)
        #     dir_cls_preds = dir_cls_preds.view(batch_size, self.mixture_component_mult * self.num_dir_bins, -1)
        #     out_tensors.update({'out_dir_cls': dir_cls_preds})

        # if self.auxiliary_corner_loss:
        # if self.training and self.auxiliary_corner_loss:
        #     out_tensor_aux = self.aux_branch.forward(fmap)
        #     out_tensor_aux = out_tensor_aux.view((batch_size, self.aux_output_ch, -1))
        #     out_tensors_aux = torch.split(out_tensor_aux, self.aux_split_indices, dim=1)
        #     out_tensors.update({'out_aux': out_tensors_aux})

        return out_tensors

    def __decode__(self, o1, center_offset):
        if self.use_corner_reg:
            o1_xy_flt, o1_z_flt, o1_xy_brb, o1_z_brb, o1_w = torch.split(o1, [2, 1, 2, 1, 1], dim=1)
            o1_xy_flt = o1_xy_flt + center_offset
            o1_xy_brb = o1_xy_brb + center_offset
            o1_w = F.relu(o1_w)
            mu = torch.cat([o1_xy_flt, o1_z_flt, o1_xy_brb, o1_z_brb, o1_w], dim=1)
        elif self.use_front_center_reg:
            o1_xy_center, o1_z_center, o1_xy_fcenter, o1_wh = torch.split(o1, [2, 1, 2, 2], dim=1)
            o1_xy_center = o1_xy_center + center_offset
            o1_xy_fcenter = o1_xy_fcenter + center_offset
            o1_wh = F.relu(o1_wh)
            mu = torch.cat([o1_xy_center, o1_z_center, o1_xy_fcenter, o1_wh], dim=1)
        else:
            o1_xy, o1_z, o1_lwh, o1_dir = torch.split(o1, [2, 1, 3, self.dir_reg_channel], dim=1)
            o1_xy = o1_xy + center_offset
            if not self.use_dir_sincos and self.use_dir_cls:
                o1_dir = F.relu(o1_dir)
            mu = torch.cat([o1_xy, o1_z, o1_lwh, o1_dir], dim=1)
        return mu

    def __decode_multivariate_sig__(self, o2):
        o2_split = list(torch.split(o2, self.o2_split_ch, dim=1))
        U, sqrt_inv_det_cov = [], []
        for i in range(len(self.multivariate_split_ch)):
            triu_indices = torch.triu_indices(self.multivariate_split_ch[i], self.multivariate_split_ch[i]).cuda()
            diag_indices = torch.arange(0, self.multivariate_split_ch[i], dtype=torch.int64).cuda()
            U.append(torch.zeros([o2_split[i].shape[0], self.multivariate_split_ch[i], self.multivariate_split_ch[i],
                                  o2_split[i].shape[-1]]).cuda())
            U[i][:, triu_indices[0], triu_indices[1], :] = o2_split[i]
            U[i][:, diag_indices, diag_indices, :] = torch.relu(U[i][:, diag_indices, diag_indices, :]) + lib_util.epsilon
            sqrt_inv_det_cov_temp = torch.prod(U[i][:, diag_indices, diag_indices, :], dim=1, keepdim=True)
            sqrt_inv_det_cov.append(sqrt_inv_det_cov_temp)

            # sqrt_inv_det_cov_temp = torch.sum(U[i][:, diag_indices, diag_indices, :], dim=1, keepdim=True)
            # U[i][:, diag_indices, diag_indices, :] = torch.exp(U[i][:, diag_indices, diag_indices, :])
            # sqrt_inv_det_cov.append(torch.exp(sqrt_inv_det_cov_temp))

        return U, sqrt_inv_det_cov

    def __decode_aux_corners__(self, out_aux_corners, center_offset):
        # center_offset_z = torch.zeros((center_offset.shape[0], 1, center_offset.shape[2]), dtype=center_offset.dtype,
        #             device=center_offset.device)
        # center_offset_xyz = torch.cat((center_offset, center_offset_z), dim=1)
        # center_offset_xyz = torch.cat([center_offset_xyz] * 8, dim=1)
        #
        # assert center_offset_xyz.shape[1] == out_aux_corners.shape[1] \
        #        and center_offset_xyz.shape[2] == out_aux_corners.shape[2]
        #
        # out_aux_corners = out_aux_corners + center_offset_xyz
        o1_xy_frt, o1_z_frt, o1_xy_blb, o1_z_blb, o1_l = torch.split(out_aux_corners, [2, 1, 2, 1, 1], dim=1)
        o1_xy_frt = o1_xy_frt + center_offset
        o1_xy_blb = o1_xy_blb + center_offset
        o1_l = F.relu(o1_l)
        mu = torch.cat([o1_xy_frt, o1_z_frt, o1_xy_blb, o1_z_blb, o1_l], dim=1)
        return mu

    def sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def __get_mixture_params__(self, out_tensors, net_data_dict):
        pi_cap_per_class = None
        # if self.use_dir_cls:
        #     out_mu, out_sig, out_pi, dir_cls_pred = out_tensors['out_reg']
        # else:
        #     out_mu, out_sig, out_pi = out_tensors['out_reg']
        # if 'out_cls' in out_tensors.keys():
        #     out_prob = out_tensors['out_cls']
        # if 'out_pi_cap_per_class' in out_tensors.keys():
        #     pi_cap_per_class = out_tensors['out_pi_cap_per_class']
        # if 'out_dir_cls' in out_tensors.keys():
        #     dir_cls_pred = out_tensors['out_dir_cls']
        # if 'out_aux' in out_tensors.keys():
        #     if self.auxiliary_share_pi:
        #         out_aux_mu, out_aux_sig = out_tensors['out_aux']
        #     else:
        #         out_aux_mu, out_aux_sig, out_aux_pi = out_tensors['out_aux']
#############
        out_mu, out_pi = out_tensors['out_reg']
        out_sig = None
        if 'out_reg_sig' in out_tensors.keys():
            out_sig = out_tensors['out_reg_sig']
        if 'out_cls' in out_tensors.keys():
            out_prob = out_tensors['out_cls']
#############
        center_offset = net_data_dict['center_offset']
        mixture_component_mult = self.mixture_component_mult
        # if self.use_multihead:
        #     mixture_component_mult = self.num_class if self.sigmoid_prob else self.num_class-1

        mu = self.__decode__(out_mu, center_offset)

        if self.multivariate:
            U, sqrt_inv_det_cov = self.__decode_multivariate_sig__(out_sig)
            sig = None
        else:
            sig = None
            if out_sig is not None:
                sig_min = torch.ones_like(mu) * lib_util.epsilon
                sig = torch.max(func.softplus(out_sig), sig_min)

        if self.sigmoid_prob:
            prob = self.sigmoid(out_prob)
        else:
            prob = func.softmax(out_prob, dim=1)

        pi = func.softmax(out_pi, dim=2)

        if self.fuse_pi_w_prob:
            # pi2 = pi.clone()
            norm_pi = (pi / torch.max(pi, dim=-1, keepdim=True)[0])
            # prob = (prob ** (1-self.fuse_alpha)) * (norm_pi ** (self.fuse_alpha))
            prob = (prob * (1 - self.fuse_alpha)) + (norm_pi * self.fuse_alpha)

        if not self.training and not self.fuse_pi_w_prob and self.infer_pi_fuse_alpha > 0.0:
            norm_pi = (pi / torch.max(pi, dim=-1, keepdim=True)[0])
            prob = (1 - self.infer_pi_fuse_alpha) * prob + self.infer_pi_fuse_alpha * norm_pi
            # prob = torch.sqrt(prob * norm_pi)

        param_dict = {'mu': mu, 'sig': sig, 'prob': prob, 'pi': pi}

        if self.multivariate:
            param_dict.update({'sqrt_inv_det_cov': sqrt_inv_det_cov, 'U': U})

        return param_dict

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2
    ):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        # ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            # ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            # ret_boxes[k, 2] = z[k]
            # ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            # ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            # ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            # if gt_boxes.shape[1] > 8:
            #     ret_boxes[k, 8:] = gt_boxes[k, 7:-1]


        return heatmap, inds, mask

    def assign_targets(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            feature_map_size: (2) [H, W]
        Returns:

        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate([self.class_names]):
            heatmap_list, target_boxes_list, inds_list, masks_list = [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, inds, mask = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
        return ret_dict

    def get_loss(self, global_step=None):
        loss_dict, value_dict = self.loss_func.forward(self.forward_ret_dict, global_step=global_step)
        mog_nll, mod_nll, aux_corners_mog_nll, topk_reg = 0, 0, 0, 0
        tb_dict = {}
        if 'mog_nll' in loss_dict.keys():
            mog_nll = loss_dict["mog_nll"]
            mog_nll = torch.sum(mog_nll, dim=0) / mog_nll.shape[0]
            tb_dict['mog_nll'] = mog_nll.item()
        if 'topk_reg' in loss_dict.keys():
            topk_reg = loss_dict['topk_reg']
            topk_reg = torch.sum(topk_reg, dim=0) / topk_reg.shape[0]
            tb_dict['topk_reg'] = topk_reg.item()
        if 'mod_nll' in loss_dict.keys():
            mod_nll = loss_dict["mod_nll"]
            mod_nll = torch.sum(mod_nll, dim=0) / self.forward_ret_dict['data_dict']['batch_size']
            # mod_nll = torch.sum(mod_nll, dim=0) / mod_nll.shape[0]
            tb_dict['mod_nll'] = mod_nll.item()
        elif self.hm_loss_func is not None:
            pred = self.forward_ret_dict['output_dict']['prob']
            target = self.forward_ret_dict['target_dicts']['heatmaps'][0]
            mod_nll = self.hm_loss_func(pred.reshape(target.shape), target)
            mod_nll *= self.model_cfg.LOSS_CONFIG.LOSS_FUNC_ARGS['lw_dict']['mod_nll']
            tb_dict['mod_nll'] = mod_nll.item()
        if 'aux_corners_mog_nll' in loss_dict.keys():
            aux_corners_mog_nll = loss_dict["aux_corners_mog_nll"]
            aux_corners_mog_nll = torch.sum(aux_corners_mog_nll, dim=0) / aux_corners_mog_nll.shape[0]
            tb_dict['aux_corners_mog_nll'] = aux_corners_mog_nll.item()

        rpn_loss = mog_nll + mod_nll + topk_reg + aux_corners_mog_nll
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, output_dict, raw_output=False):
        mu = output_dict['mu']
        prob = output_dict['prob']
        pi = output_dict['pi']
        sig = output_dict['sig']
        dir_cls_preds = None
        if 'dir_cls_preds' in output_dict.keys():
            dir_cls_preds = output_dict['dir_cls_preds']
            if len(dir_cls_preds) == 0:
                dir_cls_preds = None

        if self.remove_cls_branch:
            prob_list = []
            for i in range(len(prob)):
                prob_s = torch.zeros([prob[i].shape[0], self.num_class, prob[i].shape[2]], dtype=prob[i].dtype, device=prob[i].device)
                prob_s[:, i:i+1, :] = prob[i]
                prob_list.append(prob_s)
            prob = prob_list

        if isinstance(mu, list):
            mu = torch.cat(mu, dim=-1)
            prob = torch.cat(prob, dim=-1)
            pi = torch.cat(pi, dim=-1)
            if sig[0] is not None:
                sig = torch.cat(sig, dim=-1)
            if dir_cls_preds is not None:
                dir_cls_preds = torch.cat(dir_cls_preds, dim=-1)
            if self.shared_pi_prob:
                prob_new = torch.zeros((mu.shape[0], self.num_class, mu.shape[2]), dtype=mu.dtype, device=mu.device)
                num_k = prob.shape[-1]
                for c in range(self.num_class):
                    prob_new[:, c, num_k*c:num_k*(c+1)] = prob[:, c, :]
                prob = prob_new
                pi = torch.cat([pi] * self.num_class, dim=-1)

        batch_cls_preds = prob.transpose(1, 2)
        batch_box_preds = mu.transpose(1, 2)

        if self.use_corner_reg:
            batch_box_preds = box_utils.decode_fltbrbw_to_boxes(batch_box_preds)

        elif self.use_front_center_reg:
            batch_box_preds = box_utils.decode_cfcwh_to_boxes(batch_box_preds)

        else: # mu: (x, y, z, l, w, h, dir)
            if not self.lidar_coord_reg:
                xy_lidar_to_bev = batch_box_preds.new_tensor(np.flip(self.xy_lidar_to_bev).copy())
                point_cloud_range = batch_box_preds.new_tensor(self.point_cloud_range)
                batch_box_preds[..., :2] = batch_box_preds[..., :2] / xy_lidar_to_bev + point_cloud_range[:2]
                # batch_box_preds[..., 3:5] = batch_box_preds[..., 3:5] / xy_lidar_to_bev

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

        if raw_output:
            return batch_cls_preds, batch_box_preds

        fg_mask = torch.ones_like(pi.squeeze(dim=1), dtype=torch.bool)


        norm_pi = (pi / torch.max(pi, dim=-1, keepdim=True)[0]).squeeze(dim=1)
        if self.infer_pi_thresh > 0:
            pi_keep_mask = norm_pi > self.infer_pi_thresh
            fg_mask = fg_mask & pi_keep_mask

        if self.infer_remove_bg and not self.sigmoid_prob:
            cls_keep_mask = torch.argmax(batch_cls_preds, dim=-1) != 0
            fg_mask = fg_mask & cls_keep_mask

        if self.infer_get_local_peak:
            spatial_dim = (self.input_size * self.feature_scales[0]).astype(np.int32)
            # norm_pi = norm_pi.view(-1, 1, spatial_dim[0], spatial_dim[1])

            sig_clamp = torch.clamp(sig, min=0.01)
            inv_sig = 1.0 / sig_clamp
            norm_inv_sig = inv_sig / inv_sig.max(axis=2, keepdim=True)[0]
            mean_norm_inv_sig = norm_inv_sig.mean(axis=1, keepdim=True).squeeze(dim=1)
            reg_score = torch.sqrt(norm_pi * mean_norm_inv_sig)
            reg_score = reg_score.view(-1, 1, spatial_dim[0], spatial_dim[1])

            local_kernel_size = 3
            temp = F.max_pool2d(reg_score, kernel_size=local_kernel_size, stride=1, padding=int((local_kernel_size-1)/2))
            local_peak_mask = reg_score == temp
            reg_score = reg_score.view(reg_score.shape[0], -1)
            local_peak_mask = local_peak_mask.view(reg_score.shape[0], -1)
            fg_mask = fg_mask & local_peak_mask

        # remove illegal box
        illegal_length = 0.01
        legal_mask = (batch_box_preds[..., 3] > illegal_length) & (batch_box_preds[..., 4] > illegal_length) \
                     & (batch_box_preds[..., 5] > illegal_length)
        fg_mask = fg_mask & legal_mask

        batch_index = []
        for i in range(fg_mask.shape[0]):
            num_fg = batch_cls_preds[i][fg_mask[i]].shape[0]
            batch_index.append(torch.ones(num_fg, device=fg_mask.device) * i)
        batch_index = torch.cat(batch_index)

        if self.sigmoid_prob:
            batch_cls_preds = batch_cls_preds[fg_mask][:]
        else:
            batch_cls_preds = batch_cls_preds[fg_mask][:, 1:]
        batch_box_preds = batch_box_preds[fg_mask]

        return batch_cls_preds, batch_box_preds, batch_index, norm_pi, pi.squeeze(1), sig, prob

    def generate_predicted_boxes_fast(self, output_dict):
        mu = output_dict['mu']
        prob = output_dict['prob']
        pi = output_dict['pi']

        if isinstance(mu, list):
            mu = torch.cat(mu, dim=-1)
            prob = torch.cat(prob, dim=-1)
            pi = torch.cat(pi, dim=-1)

        batch_cls_preds = prob.transpose(1, 2)
        batch_box_preds = mu.transpose(1, 2)

        if self.use_corner_reg:
            batch_box_preds = box_utils.decode_fltbrbw_to_boxes(batch_box_preds)

        elif self.use_front_center_reg:
            batch_box_preds = box_utils.decode_cfcwh_to_boxes(batch_box_preds)

        else: # mu: (x, y, z, l, w, h, dir)
            if not self.lidar_coord_reg:
                xy_lidar_to_bev = batch_box_preds.new_tensor(np.flip(self.xy_lidar_to_bev).copy())
                point_cloud_range = batch_box_preds.new_tensor(self.point_cloud_range)
                batch_box_preds[..., :2] = batch_box_preds[..., :2] / xy_lidar_to_bev + point_cloud_range[:2]
                # batch_box_preds[..., 3:5] = batch_box_preds[..., 3:5] / xy_lidar_to_bev

            if self.use_dir_sincos:
                dir = torch.atan2(batch_box_preds[..., 6], batch_box_preds[..., 7]).unsqueeze(-1)
                batch_box_preds = torch.cat((batch_box_preds[..., :6], dir), dim=-1)

        fg_mask = torch.ones_like(pi.squeeze(dim=1), dtype=torch.bool)

        norm_pi = (pi / torch.max(pi, dim=-1, keepdim=True)[0]).squeeze(dim=1)
        if self.infer_pi_thresh > 0:
            pi_keep_mask = norm_pi > self.infer_pi_thresh
            fg_mask = fg_mask & pi_keep_mask

        # remove illegal box
        illegal_length = 0.01
        legal_mask = (batch_box_preds[..., 3] > illegal_length) & (batch_box_preds[..., 4] > illegal_length) \
                     & (batch_box_preds[..., 5] > illegal_length)
        fg_mask = fg_mask & legal_mask

        batch_index = []
        for i in range(fg_mask.shape[0]):
            num_fg = batch_cls_preds[i][fg_mask[i]].shape[0]
            batch_index.append(torch.ones(num_fg, device=fg_mask.device) * i)
        batch_index = torch.cat(batch_index)

        batch_cls_preds = batch_cls_preds[fg_mask]
        batch_box_preds = batch_box_preds[fg_mask]

        return batch_cls_preds, batch_box_preds, batch_index


    def forward(self, data_dict):
        ###################3
        # vis_spatial_feature = torch.mean(data_dict['spatial_features'], dim=1).detach().cpu().numpy()
        # vis_spatial_feature = vis_spatial_feature / np.max(vis_spatial_feature)
        # vis_spatial_feature_2d = torch.mean(data_dict['spatial_features_2d'], dim=1).detach().cpu().numpy()
        # vis_spatial_feature_2d = vis_spatial_feature_2d / np.max(vis_spatial_feature_2d)
        # import matplotlib.pyplot as plt
        # # cv2.imshow("spatial_feature", vis_spatial_feature[0])
        # plt.imshow(vis_spatial_feature_2d[0])
        # plt.show()
        ############3

        spatial_features_2d = data_dict['spatial_features_2d']
        batch_size = spatial_features_2d.shape[0]
        data_dict['batch_size'] = batch_size
        # net_data_dict = self.__sync_batch_and_device__(batch_size, spatial_features_2d.device.index)
        net_data_dict = {'center_offset': self.center_offset}

        out_tensors = self.__get_output_tensors__(spatial_features_2d, batch_size)
        output_dict = self.__get_mixture_params__(out_tensors, net_data_dict)

        if self.training:
            if self.predict_boxes_when_training:
                batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(output_dict, raw_output=True)
                data_dict['batch_cls_preds'] = batch_cls_preds
                data_dict['batch_box_preds'] = batch_box_preds
            data_dict.update(output_dict)
            self.forward_ret_dict['output_dict'] = output_dict
            self.forward_ret_dict['data_dict'] = data_dict

            return data_dict

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds, batch_index = self.generate_predicted_boxes_fast(output_dict)
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['batch_index'] = batch_index
            data_dict['cls_preds_normalized'] = True

            # batch_cls_preds, batch_box_preds, batch_index, batch_pi_norm, batch_pi, batch_sig, batch_prob = self.generate_predicted_boxes(output_dict)
            # data_dict['batch_cls_preds'] = batch_cls_preds
            # data_dict['batch_box_preds'] = batch_box_preds
            # data_dict['batch_index'] = batch_index
            # data_dict['cls_preds_normalized'] = True
            # data_dict['batch_pi_norm'] = batch_pi_norm
            # data_dict['batch_pi'] = batch_pi
            # data_dict['batch_prob'] = batch_prob
            # data_dict['batch_sig'] = batch_sig
            # if isinstance(batch_sig, list) and batch_sig[0] is None:
            #     data_dict['batch_sig'] = None

            # if 'aux_mu' in output_dict:
            #     data_dict['aux_mu'] = output_dict['aux_mu']
            #     data_dict['aux_pi'] = output_dict['aux_pi']
            return data_dict

