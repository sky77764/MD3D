import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_utils

import abc
from . import md3d_func, md3d_utils, common_utils

class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, activated=False):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        if not activated:
            pred_sigmoid = torch.sigmoid(input)
        else:
            pred_sigmoid = input
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

class WeightedClassificationLoss(nn.Module):
    def __init__(self):
        super(WeightedClassificationLoss, self).__init__()

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights=None, reduction='none'):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        if weights is not None:
            if weights.shape.__len__() == 2 or \
                    (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
                weights = weights.unsqueeze(-1)

            assert weights.shape.__len__() == bce_loss.shape.__len__()

            loss = weights * bce_loss
        else:
            loss = bce_loss

        if reduction == 'none':
            return loss
        elif reduction == 'sum':
            loss = loss.sum(dim=-1)
        elif reduction == 'mean':
            loss = loss.mean(dim=-1)
        return loss

class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)


def compute_fg_mask(gt_boxes2d, shape, downsample_factor=1, device=torch.device("cpu")):
    """
    Compute foreground mask for images
    Args:
        gt_boxes2d: (B, N, 4), 2D box labels
        shape: torch.Size or tuple, Foreground mask desired shape
        downsample_factor: int, Downsample factor for image
        device: torch.device, Foreground mask desired device
    Returns:
        fg_mask (shape), Foreground mask
    """
    fg_mask = torch.zeros(shape, dtype=torch.bool, device=device)

    # Set box corners
    gt_boxes2d /= downsample_factor
    gt_boxes2d[:, :, :2] = torch.floor(gt_boxes2d[:, :, :2])
    gt_boxes2d[:, :, 2:] = torch.ceil(gt_boxes2d[:, :, 2:])
    gt_boxes2d = gt_boxes2d.long()

    # Set all values within each box to True
    B, N = gt_boxes2d.shape[:2]
    for b in range(B):
        for n in range(N):
            u1, v1, u2, v2 = gt_boxes2d[b, n]
            fg_mask[b, v1:v2, u1:u2] = True

    return fg_mask


def neg_loss_cornernet(pred, gt, mask=None):
    """
    Refer to https://github.com/tianweiy/CenterPoint.
    Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
    Args:
        pred: (batch x c x h x w)
        gt: (batch x c x h x w)
        mask: (batch x h x w)
    Returns:
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    if mask is not None:
        mask = mask[:, None, :, :].float()
        pos_loss = pos_loss * mask
        neg_loss = neg_loss * mask
        num_pos = (pos_inds.float() * mask).sum()
    else:
        num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """
    def __init__(self):
        super(FocalLossCenterNet, self).__init__()
        self.neg_loss = neg_loss_cornernet

    def forward(self, out, target, mask=None):
        return self.neg_loss(out, target, mask=mask)


def _reg_loss(regr, gt_regr, mask):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    L1 regression loss
    Args:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    Returns:
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()
    isnotnan = (~ torch.isnan(gt_regr)).float()
    mask *= isnotnan
    regr = regr * mask
    gt_regr = gt_regr * mask

    loss = torch.abs(regr - gt_regr)
    loss = loss.transpose(2, 0)

    loss = torch.sum(loss, dim=2)
    loss = torch.sum(loss, dim=1)
    # else:
    #  # D x M x B
    #  loss = loss.reshape(loss.shape[0], -1)

    # loss = loss / (num + 1e-4)
    loss = loss / torch.clamp_min(num, min=1.0)
    # import pdb; pdb.set_trace()
    return loss


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class RegLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(RegLossCenterNet, self).__init__()

    def forward(self, output, mask, ind=None, target=None):
        """
        Args:
            output: (batch x dim x h x w) or (batch x max_objects)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """
        if ind is None:
            pred = output
        else:
            pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss


class LossFunctionABC(abc.ABC):
    def __init__(self, global_args, loss_func_args):
        self.global_args = global_args
        self.loss_func_args = loss_func_args
        self.lw_dict = loss_func_args['lw_dict']

    def update(self, new_loss_args):
        if 'lw_dict' in new_loss_args.keys():
            pre_lw_dict_str = str(self.lw_dict)
            self.lw_dict.update(new_loss_args['lw_dict'])
            print('[LOSS FUNCTION] lw dict:', pre_lw_dict_str, '->', self.lw_dict)

    @ abc.abstractmethod
    def forward(self, *x):
        pass

class MD3DLossFunc(LossFunctionABC):
    def __init__(self, global_args, loss_func_args):
        super(MD3DLossFunc, self).__init__(global_args, loss_func_args)

        self.coord_pdf = md3d_utils.gaussian_pdf
        if loss_func_args['coord_pdf'] == 'cauchy':
            self.coord_pdf = md3d_utils.cauchy_pdf
        elif loss_func_args['coord_pdf'] == 'laplace':
            self.coord_pdf = md3d_utils.laplace_pdf

        self.multivariate = loss_func_args['multivariate']
        if self.multivariate:
            self.coord_pdf = md3d_utils.multivariate_gaussian_pdf
        self.separate_classes_ch = loss_func_args['separate_classes_ch']

        self.mog_pi_thresh = loss_func_args['mog_pi_thresh']
        self.mog_topk_comp = loss_func_args['mog_topk_comp'] if 'mog_topk_comp' in loss_func_args else 0
        self.mod_pi_thresh = loss_func_args['mod_pi_thresh']

        self.value_return = loss_func_args['value_return']
        self.n_classes = global_args['n_classes']

        self.use_dir_sincos = loss_func_args['use_dir_sincos']
        self.lidar_coord_reg = loss_func_args['lidar_coord_reg'] if 'lidar_coord_reg' in loss_func_args else True

        self.sigmoid_prob = loss_func_args['sigmoid_prob']
        self.fcos_like_cls = loss_func_args['fcos_like_cls'] if 'fcos_like_cls' in loss_func_args else False
        self.origin_cls_step = loss_func_args['origin_cls_step'] if 'origin_cls_step' in loss_func_args else -1
        self.centerpoint_cls = loss_func_args['centerpoint_cls'] if 'centerpoint_cls' in loss_func_args else False

        # self.dir_offset = loss_func_args['dir_offset']
        # self.dir_limit_offset = loss_func_args['dir_limit_offset']
        # self.num_dir_bins = loss_func_args['num_dir_bins']

        self.use_corner_reg = loss_func_args['use_corner_reg']
        self.use_front_center_reg = loss_func_args['use_front_center_reg'] if 'use_front_center_reg' in loss_func_args else False

        assert 'mog_nll' in self.lw_dict.keys()
        assert 'mod_nll' in self.lw_dict.keys()
        assert loss_func_args['coord_pdf'] in ('gaussian', 'cauchy', 'laplace')

    def forward(self, forward_ret_dict, global_step=None):
        output_dict = forward_ret_dict['output_dict']
        data_dict = forward_ret_dict['data_dict']
        point_cloud_range = forward_ret_dict['point_cloud_range']
        coord_range = forward_ret_dict['coord_range']
        xy_lidar_to_bev = np.flip(forward_ret_dict['xy_lidar_to_bev'])

        mu, sig = output_dict['mu'], output_dict['sig']
        prob, pi = output_dict['prob'], output_dict['pi']
        boxes, labels = data_dict['gt_boxes'][..., :-1], data_dict['gt_boxes'][..., -1:].to(torch.int64)
        n_boxes = (labels != 0).sum(dim=1).squeeze(-1)

        U, sqrt_inv_det_cov = None, None
        if self.multivariate:
            U, sqrt_inv_det_cov = output_dict['U'], output_dict['sqrt_inv_det_cov']

        zero_box = False
        if torch.sum(n_boxes) == 0:
            n_boxes[0] += 1
            zero_box = True

        if not self.lidar_coord_reg and not self.use_corner_reg:
            point_cloud_range = boxes.new_tensor(point_cloud_range)
            xy_lidar_to_bev = boxes.new_tensor(xy_lidar_to_bev.copy())
            boxes[..., :2] = (boxes[..., :2] - point_cloud_range[:2]) * xy_lidar_to_bev
            # boxes[..., 3:5] = boxes[..., 3:5] * xy_lidar_to_bev

        dir_cls_preds, dir_cls_targets = None, None
        # if 'dir_cls_preds' in output_dict.keys() and not self.use_corner_reg and not self.use_front_center_reg:
        #     dir_cls_preds = output_dict['dir_cls_preds']
        #     dir_cls_targets, dir_res_targets = get_direction_cls_target(boxes, dir_offset=self.dir_offset, num_bins=self.num_dir_bins)
        #     boxes[..., 6] = dir_res_targets

        if self.use_corner_reg:
            boxes, aux_corners = box_utils.encode_boxes_to_fltbrbw(boxes,
                                        return_aux_corners=self.lw_dict['aux_corners_mog_nll'] > 0)

        elif self.use_front_center_reg:
            boxes, aux_corners = box_utils.encode_boxes_to_cfcwh(boxes,
                                        return_all_corners=self.lw_dict['aux_corners_mog_nll'] > 0)

        if self.fcos_like_cls and global_step is not None and global_step > self.origin_cls_step:
            self.fcos_like_cls = False

        loss_dict = {}
        value_dict = {}
        if self.lw_dict['mog_nll'] > 0:
            if isinstance(mu, list):
                max_nll, topk_reg = [], []
                for i in range(len(mu)):
                    out_loss_dict = md3d_func.calc_mog_nll(
                        mu[i], sig[i], pi[i] if len(pi) > 1 else pi[0], boxes, labels, n_boxes, self.coord_pdf,
                        self.mog_pi_thresh, self.value_return,
                        use_dir_sincos=self.use_dir_sincos, U=U[i] if U is not None else None,
                        sqrt_inv_det_cov=sqrt_inv_det_cov[i] if sqrt_inv_det_cov is not None else None,
                        separate_classes_ch=self.separate_classes_ch, cur_cls_idx=i,
                        dir_cls_preds=dir_cls_preds[i] if dir_cls_preds is not None and len(dir_cls_preds) > 0 else None, dir_cls_targets=dir_cls_targets,
                        topk_comp=self.mog_topk_comp)
                    max_nll.append(out_loss_dict['mog_nll_loss'])
                    if 'topk_reg' in out_loss_dict.keys():
                        topk_reg.append(out_loss_dict['topk_reg'])

                max_nll = torch.cat(max_nll)
                if len(topk_reg) > 0:
                    topk_reg = torch.cat(topk_reg)
            else:
                out_loss_dict = md3d_func.calc_mog_nll(
                    mu, sig, pi, boxes, labels, n_boxes, self.coord_pdf, self.mog_pi_thresh, self.value_return,
                    use_dir_sincos=self.use_dir_sincos, U=U, sqrt_inv_det_cov=sqrt_inv_det_cov,
                    separate_classes_ch=self.separate_classes_ch,
                    dir_cls_preds=dir_cls_preds, dir_cls_targets=dir_cls_targets,
                    topk_comp=self.mog_topk_comp)
                max_nll = out_loss_dict['mog_nll_loss']
                if 'topk_reg' in out_loss_dict.keys():
                    topk_reg = out_loss_dict['topk_reg']

            max_nll = max_nll[~torch.isnan(max_nll)]
            loss_dict.update({'mog_nll': self.lw_dict['mog_nll'] * max_nll})

            if 'topk_reg' in self.lw_dict and self.lw_dict['topk_reg'] > 0 and self.mog_topk_comp > 0:
                topk_reg = topk_reg[~torch.isnan(topk_reg)]
                loss_dict.update({'topk_reg': self.lw_dict['topk_reg'] * topk_reg})

        if self.lw_dict['mod_nll'] > 0 and not self.centerpoint_cls:
            matched_threshold, unmatched_threshold = None, None
            if 'matched_threshold' in self.loss_func_args.keys():
                matched_threshold = self.loss_func_args['matched_threshold']
                unmatched_threshold = self.loss_func_args['unmatched_threshold']
            iou_based_cls_target = False
            if 'iou_based_cls_target' in self.loss_func_args.keys():
                iou_based_cls_target = self.loss_func_args['iou_based_cls_target']
            if isinstance(mu, list):
                mod_nll_loss_return = []
                for i in range(len(mu)):
                    mod_nll_loss_return.append(md3d_func.calc_mod_focal_loss(
                        mu[i].detach(), sig[i].detach() if sig[i] is not None else None,
                        pi[i].detach() if len(pi) > 1 else pi[0].detach(), prob[i] if len(prob) > 1 else prob[0],
                        boxes, labels, n_boxes,
                        self.mod_pi_thresh, self.n_classes, use_dir_sincos=self.use_dir_sincos,
                        sigmoid_prob=self.sigmoid_prob,
                        separate_classes_ch=self.separate_classes_ch, cur_cls_idx=i,
                        use_corner_reg=self.use_corner_reg, use_front_center_reg=self.use_front_center_reg,
                        fcos_like_cls=self.fcos_like_cls, centerpoint_cls=self.centerpoint_cls,
                        hm_target=forward_ret_dict['target_dicts']['heatmaps'][0] if 'target_dicts' in forward_ret_dict.keys() else None,
                        matched_threshold=matched_threshold, unmatched_threshold=unmatched_threshold,
                        iou_based_cls_target=iou_based_cls_target))
                mod_nll_loss_return = torch.cat(mod_nll_loss_return)
            else:
                mod_nll_loss_return = md3d_func.calc_mod_focal_loss(
                    mu.detach(), sig.detach() if sig is not None else None, pi.detach(), prob, boxes, labels,
                    n_boxes,
                    self.mod_pi_thresh, self.n_classes, use_dir_sincos=self.use_dir_sincos,
                    sigmoid_prob=self.sigmoid_prob,
                    use_corner_reg=self.use_corner_reg, use_front_center_reg=self.use_front_center_reg,
                    fcos_like_cls=self.fcos_like_cls, centerpoint_cls=self.centerpoint_cls,
                    hm_target=forward_ret_dict['target_dicts']['heatmaps'][0] if 'target_dicts' in forward_ret_dict.keys() else None,
                    matched_threshold=matched_threshold, unmatched_threshold=unmatched_threshold,
                    iou_based_cls_target=iou_based_cls_target)

            mod_nll_loss_return = mod_nll_loss_return[~torch.isnan(mod_nll_loss_return)]
            loss_dict.update({'mod_nll': self.lw_dict['mod_nll'] * mod_nll_loss_return})

        if zero_box:
            loss_dict['mog_nll'][0] *= 1e-30

        if self.lw_dict['aux_corners_mog_nll'] > 0:
            aux_mu, aux_sig, aux_pi = output_dict['aux_mu'], output_dict['aux_sig'], output_dict['aux_pi']
            out_loss_dict = md3d_func.calc_mog_nll(
                aux_mu, aux_sig, aux_pi if aux_pi is not None else pi, aux_corners, None, n_boxes, self.coord_pdf, self.mog_pi_thresh, self.value_return)
            aux_corners_mog_nll_return = out_loss_dict['mog_nll_loss']

            loss_dict.update({'aux_corners_mog_nll': self.lw_dict['aux_corners_mog_nll'] * aux_corners_mog_nll_return})
            if zero_box:
                loss_dict['aux_corners_mog_nll'][0] *= 1e-30

        return loss_dict, value_dict

class MD3DLossFunc_centerpoint(LossFunctionABC):
    def __init__(self, global_args, loss_func_args):
        super(MD3DLossFunc_centerpoint, self).__init__(global_args, loss_func_args)

        self.coord_pdf = md3d_utils.gaussian_pdf
        if loss_func_args['coord_pdf'] == 'cauchy':
            self.coord_pdf = md3d_utils.cauchy_pdf
        elif loss_func_args['coord_pdf'] == 'laplace':
            self.coord_pdf = md3d_utils.laplace_pdf

        self.multivariate = False
        if self.multivariate:
            self.coord_pdf = md3d_utils.multivariate_gaussian_pdf
        self.separate_classes_ch = False

        self.mog_pi_thresh = loss_func_args['mog_pi_thresh']
        self.mog_topk_comp = 0
        self.mod_pi_thresh = loss_func_args['mod_pi_thresh']

        self.value_return = False
        self.n_classes = global_args['n_classes']

        self.use_dir_sincos = False
        self.lidar_coord_reg = True

        self.sigmoid_prob = True
        self.fcos_like_cls = False
        self.origin_cls_step = -1
        self.centerpoint_cls = False

        self.use_corner_reg = True
        self.use_front_center_reg = False

        assert 'mog_nll' in self.lw_dict.keys()
        assert 'mod_nll' in self.lw_dict.keys()
        assert loss_func_args['coord_pdf'] in ('gaussian', 'cauchy', 'laplace')

    def forward(self, forward_ret_dict, global_step=None):
        output_dict = forward_ret_dict['pred_dicts'][0]
        data_dict = forward_ret_dict['data_dict']
        # point_cloud_range = forward_ret_dict['point_cloud_range']
        # coord_range = forward_ret_dict['coord_range']
        # xy_lidar_to_bev = np.flip(forward_ret_dict['xy_lidar_to_bev'])

        mu, sig, pi = output_dict['mu'], output_dict['sig'], output_dict['pi']
        prob = None
        boxes, labels = data_dict['gt_boxes'][..., :-1], data_dict['gt_boxes'][..., -1:].to(torch.int64)
        n_boxes = (labels != 0).sum(dim=1).squeeze(-1)

        U, sqrt_inv_det_cov = None, None
        if self.multivariate:
            U, sqrt_inv_det_cov = output_dict['U'], output_dict['sqrt_inv_det_cov']

        zero_box = False
        if torch.sum(n_boxes) == 0:
            n_boxes[0] += 1
            zero_box = True

        # if not self.lidar_coord_reg and not self.use_corner_reg:
        #     point_cloud_range = boxes.new_tensor(point_cloud_range)
        #     xy_lidar_to_bev = boxes.new_tensor(xy_lidar_to_bev.copy())
        #     boxes[..., :2] = (boxes[..., :2] - point_cloud_range[:2]) * xy_lidar_to_bev
            # boxes[..., 3:5] = boxes[..., 3:5] * xy_lidar_to_bev

        dir_cls_preds, dir_cls_targets = None, None
        # if 'dir_cls_preds' in output_dict.keys() and not self.use_corner_reg and not self.use_front_center_reg:
        #     dir_cls_preds = output_dict['dir_cls_preds']
        #     dir_cls_targets, dir_res_targets = get_direction_cls_target(boxes, dir_offset=self.dir_offset, num_bins=self.num_dir_bins)
        #     boxes[..., 6] = dir_res_targets

        if self.use_corner_reg:
            boxes, _ = box_utils.encode_boxes_to_fltbrbw(boxes)

        elif self.use_front_center_reg:
            boxes, _ = box_utils.encode_boxes_to_cfcwh(boxes)

        if self.fcos_like_cls and global_step is not None and global_step > self.origin_cls_step:
            self.fcos_like_cls = False

        loss_dict = {}
        value_dict = {}
        if self.lw_dict['mog_nll'] > 0:
            if isinstance(mu, list):
                max_nll, topk_reg = [], []
                for i in range(len(mu)):
                    out_loss_dict = md3d_func.calc_mog_nll(
                        mu[i], sig[i], pi[i] if len(pi) > 1 else pi[0], boxes, labels, n_boxes, self.coord_pdf,
                        self.mog_pi_thresh, self.value_return,
                        use_dir_sincos=self.use_dir_sincos, U=U[i] if U is not None else None,
                        sqrt_inv_det_cov=sqrt_inv_det_cov[i] if sqrt_inv_det_cov is not None else None,
                        separate_classes_ch=self.separate_classes_ch, cur_cls_idx=i,
                        dir_cls_preds=dir_cls_preds[i] if dir_cls_preds is not None and len(dir_cls_preds) > 0 else None, dir_cls_targets=dir_cls_targets,
                        topk_comp=self.mog_topk_comp)
                    max_nll.append(out_loss_dict['mog_nll_loss'])
                    if 'topk_reg' in out_loss_dict.keys():
                        topk_reg.append(out_loss_dict['topk_reg'])

                max_nll = torch.cat(max_nll)
                if len(topk_reg) > 0:
                    topk_reg = torch.cat(topk_reg)
            else:
                out_loss_dict = md3d_func.calc_mog_nll(
                    mu, sig, pi, boxes, labels, n_boxes, self.coord_pdf, self.mog_pi_thresh, self.value_return,
                    use_dir_sincos=self.use_dir_sincos, U=U, sqrt_inv_det_cov=sqrt_inv_det_cov,
                    separate_classes_ch=self.separate_classes_ch,
                    dir_cls_preds=dir_cls_preds, dir_cls_targets=dir_cls_targets,
                    topk_comp=self.mog_topk_comp)
                max_nll = out_loss_dict['mog_nll_loss']
                if 'topk_reg' in out_loss_dict.keys():
                    topk_reg = out_loss_dict['topk_reg']

            max_nll = max_nll[~torch.isnan(max_nll)]
            loss_dict.update({'mog_nll': self.lw_dict['mog_nll'] * max_nll})

            if 'topk_reg' in self.lw_dict and self.lw_dict['topk_reg'] > 0 and self.mog_topk_comp > 0:
                topk_reg = topk_reg[~torch.isnan(topk_reg)]
                loss_dict.update({'topk_reg': self.lw_dict['topk_reg'] * topk_reg})

        if self.lw_dict['mod_nll'] > 0 and not self.centerpoint_cls:
            matched_threshold, unmatched_threshold = None, None
            if 'matched_threshold' in self.loss_func_args.keys():
                matched_threshold = self.loss_func_args['matched_threshold']
                unmatched_threshold = self.loss_func_args['unmatched_threshold']
            iou_based_cls_target = False
            if 'iou_based_cls_target' in self.loss_func_args.keys():
                iou_based_cls_target = self.loss_func_args['iou_based_cls_target']
            if isinstance(mu, list):
                mod_nll_loss_return = []
                for i in range(len(mu)):
                    mod_nll_loss_return.append(md3d_func.calc_mod_focal_loss(
                        mu[i].detach(), sig[i].detach() if sig[i] is not None else None,
                        pi[i].detach() if len(pi) > 1 else pi[0].detach(), prob[i] if len(prob) > 1 else prob[0],
                        boxes, labels, n_boxes,
                        self.mod_pi_thresh, self.n_classes, use_dir_sincos=self.use_dir_sincos,
                        sigmoid_prob=self.sigmoid_prob,
                        separate_classes_ch=self.separate_classes_ch, cur_cls_idx=i,
                        use_corner_reg=self.use_corner_reg, use_front_center_reg=self.use_front_center_reg,
                        fcos_like_cls=self.fcos_like_cls, centerpoint_cls=self.centerpoint_cls,
                        hm_target=forward_ret_dict['target_dicts']['heatmaps'][0] if 'target_dicts' in forward_ret_dict.keys() else None,
                        matched_threshold=matched_threshold, unmatched_threshold=unmatched_threshold,
                        iou_based_cls_target=iou_based_cls_target))
                mod_nll_loss_return = torch.cat(mod_nll_loss_return)
            else:
                mod_nll_loss_return = md3d_func.calc_mod_focal_loss(
                    mu.detach(), sig.detach() if sig is not None else None, pi.detach(), prob, boxes, labels,
                    n_boxes,
                    self.mod_pi_thresh, self.n_classes, use_dir_sincos=self.use_dir_sincos,
                    sigmoid_prob=self.sigmoid_prob,
                    use_corner_reg=self.use_corner_reg, use_front_center_reg=self.use_front_center_reg,
                    fcos_like_cls=self.fcos_like_cls, centerpoint_cls=self.centerpoint_cls,
                    hm_target=forward_ret_dict['target_dicts']['heatmaps'][0] if 'target_dicts' in forward_ret_dict.keys() else None,
                    matched_threshold=matched_threshold, unmatched_threshold=unmatched_threshold,
                    iou_based_cls_target=iou_based_cls_target)

            mod_nll_loss_return = mod_nll_loss_return[~torch.isnan(mod_nll_loss_return)]
            loss_dict.update({'mod_nll': self.lw_dict['mod_nll'] * mod_nll_loss_return})

        if zero_box:
            loss_dict['mog_nll'][0] *= 1e-30

        return loss_dict, value_dict


class MD3DPointLossFunc(LossFunctionABC):
    def __init__(self, global_args, loss_func_args):
        super(MD3DPointLossFunc, self).__init__(global_args, loss_func_args)
        self.coord_pdf = md3d_utils.cauchy_pdf \
            if loss_func_args['coord_pdf'] == 'cauchy' \
            else md3d_utils.gaussian_pdf

        self.multivariate = loss_func_args['multivariate']
        if self.multivariate:
            self.coord_pdf = md3d_utils.multivariate_gaussian_pdf

        self.mog_pi_thresh = loss_func_args['mog_pi_thresh']
        self.mod_pi_thresh = loss_func_args['mod_pi_thresh']
        self.mod_n_samples = loss_func_args['mod_n_samples']
        self.mod_max_samples = loss_func_args['mod_max_samples']

        self.sampling_noise = loss_func_args['sampling_noise']
        self.xy_noise_mult = loss_func_args['xy_noise_mult']
        self.value_return = loss_func_args['value_return']
        self.n_classes = global_args['n_classes']

        self.use_dir_sincos = loss_func_args['use_sincos_dir']

        self.mod_focal_loss = loss_func_args['mod_focal_loss']
        self.sigmoid_prob = loss_func_args['sigmoid_prob']

        # self.dir_offset = loss_func_args['dir_offset']
        self.dir_limit_offset = loss_func_args['dir_limit_offset']
        self.num_dir_bins = loss_func_args['num_dir_bins']

        self.use_corner_reg = loss_func_args['use_corner_reg']

        assert 'mog_nll' in self.lw_dict.keys()
        assert 'mod_nll' in self.lw_dict.keys()
        assert loss_func_args['coord_pdf'] in ('gaussian', 'cauchy')

    def forward(self, forward_ret_dict):
        output_dict = forward_ret_dict['output_dict']
        data_dict = forward_ret_dict['data_dict']

        mu, sig, pi = output_dict['mu'], output_dict['sig'], output_dict['pi']
        if 'prob' in output_dict.keys():
            prob = output_dict['prob']

        U, sqrt_inv_det_cov = None, None
        if self.multivariate:
            U, sqrt_inv_det_cov = output_dict['U'], output_dict['sqrt_inv_det_cov']

        if len(mu.shape) == 2:
            mu = mu.reshape([data_dict['batch_size'], -1, mu.shape[-1]]).transpose(1, 2)
        if sig is not None and len(sig.shape) == 2:
            sig = sig.reshape([data_dict['batch_size'], -1, sig.shape[-1]]).transpose(1, 2)
        if len(pi.shape) == 2:
            pi = pi.reshape([data_dict['batch_size'], -1, pi.shape[-1]]).transpose(1, 2)
        if U is not None:
            for i in range(len(U)):
                U[i] = U[i].reshape([data_dict['batch_size'], -1, U[i].shape[1], U[i].shape[1]]).transpose(1, 2).transpose(2, 3)
                sqrt_inv_det_cov[i] = sqrt_inv_det_cov[i].reshape([data_dict['batch_size'], 1, -1])

        boxes, labels = data_dict['gt_boxes'][..., :-1], data_dict['gt_boxes'][..., -1:].to(torch.int64)
        n_boxes = (labels != 0).sum(dim=1).squeeze(-1)
        # boxes, labels = data_dict['boxes'], data_dict['labels']
        # n_boxes = data_dict['n_boxes']

        dir_cls_preds, dir_cls_targets = None, None
        # if 'dir_cls_preds' in output_dict.keys() and not self.use_corner_reg:
        #     dir_cls_preds = output_dict['dir_cls_preds']
        #     if len(dir_cls_preds.shape) == 2:
        #         dir_cls_preds = dir_cls_preds.reshape([data_dict['batch_size'], -1, dir_cls_preds.shape[-1]]).transpose(1, 2)
        #     dir_cls_targets, dir_res_targets = get_direction_cls_target(boxes, dir_offset=self.dir_offset,
        #                                                     num_bins=self.num_dir_bins)
        #     boxes[..., 6] = dir_res_targets

        if self.use_corner_reg:
            boxes, _ = box_utils.encode_boxes_to_fltbrbw(boxes)

        zero_box = False
        if torch.sum(n_boxes) == 0:
            n_boxes[0] += 1
            zero_box = True
        loss_dict = {}
        value_dict = {}
        if self.lw_dict['mog_nll'] > 0:
            out_loss_dict = md3d_func.calc_mog_nll(
                mu, sig, pi, boxes, labels, n_boxes, self.coord_pdf, self.mog_pi_thresh, self.value_return,
                use_dir_sincos=self.use_dir_sincos, U=U, sqrt_inv_det_cov=sqrt_inv_det_cov,
                dir_cls_preds=dir_cls_preds, dir_cls_targets=dir_cls_targets)
            max_nll = out_loss_dict['mog_nll_loss']
            loss_dict.update({'mog_nll': self.lw_dict['mog_nll'] * max_nll})

        if self.lw_dict['mod_nll'] > 0:
            mod_nll_loss_return = md3d_func.calc_mod_focal_loss(
                mu.detach(), sig.detach(), pi.detach(), prob, boxes, labels, n_boxes,
                self.mod_pi_thresh, self.n_classes, use_dir_sincos=self.use_dir_sincos,
                sigmoid_prob=self.sigmoid_prob,
                use_corner_reg=self.use_corner_reg)
            mod_nll_loss_return = mod_nll_loss_return[~torch.isnan(mod_nll_loss_return)]
            loss_dict.update({'mod_nll': self.lw_dict['mod_nll'] * mod_nll_loss_return})

        if zero_box:
            loss_dict['mog_nll'][0] *= 1e-30

        return loss_dict, value_dict