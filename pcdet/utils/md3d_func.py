import copy
import time

import numpy as np
import torch
import torch.nn.functional as func

from . import loss_utils, box_utils
from . import md3d_utils as lib_util
from ..ops.iou3d_nms import iou3d_nms_utils


def focal_loss(input, target, pi=None, alpha=0.25, gamma=2.0, reduction='mean'):
    # compute the actual focal loss
    weight = torch.pow(-input + 1.0, gamma)

    if pi is None:
        focal = -alpha * weight * torch.log(input)
    else:
        if len(pi.shape) == 1:
            pi = pi.unsqueeze(-1)
        focal = -alpha * pi * weight * torch.log(input)
    loss_tmp = torch.sum(target * focal, dim=-1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


def calc_mog_nll(mu, sig, pi, boxes, labels, n_boxes, coord_pdf=lib_util.gaussian_pdf, pi_thresh=None,
                 value_return=False, use_dir_sincos=False, U=None, sqrt_inv_det_cov=None, separate_classes_ch=False,
                 cur_cls_idx=None, dir_cls_preds=None, dir_cls_targets=None, topk_comp=0):
    mog_nll_loss = list()
    pred_gt_ratio = list()
    topk_reg = list()
    for i in range(mu.shape[0]):
        if n_boxes[i] == 0:
            pass
        else:
            mu_s, pi_s = mu[i:i + 1], pi[i:i + 1]
            boxes_s = boxes[i:i + 1, :n_boxes[i]]
            if labels is not None:
                labels_s = labels[i:i + 1, :n_boxes[i]]
            U_s, sqrt_inv_det_cov_s = None, None

            sig_s = None
            if U is None:
                sig_s = sig[i:i + 1]
            else:
                U_s, sqrt_inv_det_cov_s = [], []
                for idx in range(len(U)):
                    U_s.append(U[idx][i: i + 1])
                    sqrt_inv_det_cov_s.append(sqrt_inv_det_cov[idx][i: i + 1])

            if dir_cls_preds is not None:
                dir_cls_preds_s = dir_cls_preds[i]
                dir_cls_targets_s = dir_cls_targets[i, :n_boxes[i]]

            if use_dir_sincos:
                dir_sin = torch.sin(boxes_s[..., 6:7])
                dir_cos = torch.cos(boxes_s[..., 6:7])
                boxes_s = torch.cat((boxes_s[..., :6], dir_sin, dir_cos), dim=-1)

            if pi_thresh is not None:
                max_pi_s = torch.max(pi_s)
                norm_pi_s = pi_s / max_pi_s
                keep_indices = torch.nonzero(norm_pi_s[0, 0] > pi_thresh).view(-1)
                mu_s = mu_s[:, :, keep_indices]
                if U is None:
                    sig_s = sig_s[:, :, keep_indices]
                else:
                    for idx in range(len(U_s)):
                        U_s[idx] = U_s[idx][..., keep_indices]
                        sqrt_inv_det_cov_s[idx] = sqrt_inv_det_cov_s[idx][..., keep_indices]
                if dir_cls_preds is not None:
                    dir_cls_preds_s = dir_cls_preds_s[:, keep_indices]
                pi_s = pi_s[:, :, keep_indices]
                pi_s = pi_s / torch.sum(pi_s)


            get_log_prob = False
            if separate_classes_ch and cur_cls_idx is not None:
                cls_mask = (labels_s == (cur_cls_idx + 1)).squeeze(-1)
                boxes_s = boxes_s[cls_mask].unsqueeze(0)
                if dir_cls_preds is not None:
                    dir_cls_targets_s = dir_cls_targets_s[cls_mask.squeeze(0)]

            mixture_lhs_s = lib_util.mm_pdf(
                mu_s, sig_s, pi_s, boxes_s, comp_pdf=coord_pdf, sum_comp=False,
                U=U_s, sqrt_inv_det_cov=sqrt_inv_det_cov_s, get_log_prob=get_log_prob)[0, :, 0]

            if topk_comp > 0:
                topk_values = torch.topk(mixture_lhs_s, k=topk_comp, dim=-1).values
                topk_sum = torch.sum(topk_values, dim=-1)

            if dir_cls_preds is not None:
                cat_probs_s = lib_util.category_pmf_s(dir_cls_preds_s, dir_cls_targets_s.float())
                mixture_lhs_s = mixture_lhs_s * cat_probs_s
            mixture_lhs_s = torch.sum(mixture_lhs_s, dim=-1)

            if topk_comp > 0:
                topk_reg.append(-torch.log(topk_sum / mixture_lhs_s + lib_util.epsilon))

            if get_log_prob:
                mixture_nll_s = -mixture_lhs_s
            else:
                mixture_nll_s = -torch.log(mixture_lhs_s + lib_util.epsilon)

            mog_nll_loss.append(mixture_nll_s)
            del mixture_lhs_s

            if value_return:
                norm_pi_s = pi_s / torch.max(pi_s)
                keep_indices = torch.nonzero(norm_pi_s[0, 0] > 0.001).view(-1)
                pred_gt_ratio.append((len(keep_indices) / n_boxes[i].float()).view(1))

    mog_nll_loss = torch.cat(mog_nll_loss, dim=0)
    if topk_comp > 0:
        topk_reg = torch.cat(topk_reg, dim=0)
        return {'mog_nll_loss': mog_nll_loss, "topk_reg": topk_reg}
    else:
        return {'mog_nll_loss': mog_nll_loss}


def calc_mod_focal_loss(mu, sig, pi, prob, boxes, labels, n_boxes, pi_thresh, n_classes, use_dir_sincos,
                        sigmoid_prob=False, separate_classes_ch=False, cur_cls_idx=None, shared_pi_prob=False,
                        sample_boxes_from_gaussian=True, use_corner_reg=False, use_front_center_reg=False,
                        fcos_like_cls=False, centerpoint_cls=False, hm_target=None,
                        matched_threshold=None, unmatched_threshold=None, iou_based_cls_target=False):
    # cmps_pi = pi * n_boxes.view(n_boxes.shape[0], 1, 1)
    num_mixture_components = prob.shape[-1]
    bg_labels = torch.zeros((num_mixture_components, n_classes)).float().cuda()
    if not sigmoid_prob:
        bg_labels[:, 0] = 1.0
    else:
        sigmoid_focal_loss = loss_utils.SigmoidFocalClassificationLoss()

    prob = prob.transpose(1, 2).contiguous()

    if sample_boxes_from_gaussian and sig is not None:
        pred_boxes = torch.normal(mu, sig).transpose(1, 2)
    else:
        pred_boxes = mu.transpose(1, 2)

    if use_dir_sincos and not use_corner_reg and not use_front_center_reg:
        dir = torch.atan2(pred_boxes[..., 6], pred_boxes[..., 7]).unsqueeze(-1)
        pred_boxes = torch.cat((pred_boxes[..., :6], dir), dim=-1)

    if use_corner_reg:
        pred_boxes = box_utils.decode_fltbrbw_to_boxes(pred_boxes)
        boxes = box_utils.decode_fltbrbw_to_boxes(boxes)
    elif use_front_center_reg:
        pred_boxes = box_utils.decode_cfcwh_to_boxes(pred_boxes)
        boxes = box_utils.decode_cfcwh_to_boxes(boxes)

    mod_loss = list()
    for bs_idx, (prob_s, pi_s, boxes_s, labels_s, pred_boxes_s, n_boxes_s) in \
            enumerate(zip(prob, pi, boxes, labels, pred_boxes, n_boxes)):
        if n_boxes_s <= 0:
            continue
        boxes_s, labels_s = boxes_s[:n_boxes_s], labels_s[:n_boxes_s]

        if not fcos_like_cls:
            if sigmoid_prob:
                onehot_labels_s = func.one_hot(labels_s[:, 0] - 1, n_classes).float()
            else:
                onehot_labels_s = func.one_hot(labels_s[:, 0], n_classes).float()
        bg_labels_s = bg_labels.clone()

        if pi_thresh is not None:
            max_pi_s = torch.max(pi_s)
            norm_pi_s = pi_s / max_pi_s

            keep_indices = torch.nonzero(norm_pi_s[0] > pi_thresh).view(-1)
            if keep_indices.shape[0] == 0:
                continue
            pred_boxes_s = pred_boxes_s[keep_indices, :]
            prob_s = prob_s[keep_indices, :]
            bg_labels_s = bg_labels_s[keep_indices, :]
            norm_pi_s = norm_pi_s[0, keep_indices]

        if separate_classes_ch:
            cls_mask = (labels_s == (cur_cls_idx + 1)).squeeze(-1)
            boxes_s = boxes_s[cls_mask]
            labels_s = labels_s[cls_mask]
            onehot_labels_s = onehot_labels_s[cls_mask]
            if boxes_s.shape[0] == 0:
                mod_loss.append(torch.zeros_like(prob_s))
                continue

        cls_weights_s = None
        if fcos_like_cls:
            cls_labels_s = loss_utils.get_fcos_like_cls_target(boxes_s, labels_s)
            cls_labels_s = cls_labels_s.reshape(-1, n_classes)
            if pi_thresh is not None:
                cls_labels_s = cls_labels_s[keep_indices, :]
        elif centerpoint_cls:
            cls_labels_s = hm_target[bs_idx].transpose(0, 1).transpose(1, 2)
            cls_labels_s = cls_labels_s.reshape(-1, n_classes)
            if pi_thresh is not None:
                cls_labels_s = cls_labels_s[keep_indices, :]
        else:
            if matched_threshold is None:
                iou_pairs = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes_s, boxes_s)
                max_ious, argmax_ious = torch.max(iou_pairs, dim=1)
                cls_labels_s = onehot_labels_s[argmax_ious]

                if iou_based_cls_target:
                    max_ious = max_ious.unsqueeze(dim=1)
                    positives = max_ious > 0.75
                    negatives = max_ious < 0.25
                    cls_labels_s = torch.where(negatives, bg_labels_s, cls_labels_s)
                    cls_labels_s = torch.where(positives, cls_labels_s, torch.clamp_min(2 * cls_labels_s * max_ious - 0.5, 0.0))
                else:
                    positives = max_ious.unsqueeze(dim=1) > 0.5
                    cls_labels_s = torch.where(positives, cls_labels_s, bg_labels_s)
            else:
                cls_labels_s = assign_cls_targets(pred_boxes_s, boxes_s, labels_s,
                                   matched_threshold=matched_threshold, unmatched_threshold=unmatched_threshold)
                cls_labels_s, cls_weights_s = get_cls_onehot_targets_weights(prob_s, cls_labels_s, n_classes)

        if sigmoid_prob:
            if cls_weights_s is None:
                cls_weights_s = torch.ones(cls_labels_s.shape[0], dtype=torch.float32, device=cls_labels_s.device) / cls_labels_s.shape[0]
                # cls_weights_s = torch.ones(cls_labels_s.shape[0], dtype=torch.float32, device=cls_labels_s.device) / pos_normalizer
            if shared_pi_prob:
                prob_s = prob_s[:, cur_cls_idx:cur_cls_idx+1]
                cls_labels_s = cls_labels_s[:, cur_cls_idx:cur_cls_idx + 1]
            loss = sigmoid_focal_loss(prob_s, cls_labels_s, weights=cls_weights_s, activated=True)
        else:
            loss = focal_loss(prob_s+lib_util.epsilon, cls_labels_s, reduction='none')
        mod_loss.append(loss)

    mod_loss = torch.cat(mod_loss, dim=0)
    return mod_loss

def calc_max_nll(mu, sig, pi, boxes, n_boxes):
    mog_nll_loss = list()
    for i in range(mu.shape[0]):
        if n_boxes[i] <= 0:
            pass
        else:
            mu_s, sig_s, pi_s = mu[i:i + 1], sig[i:i + 1], pi[i:i + 1]
            boxes_s = boxes[i:i + 1, :n_boxes[i]]

            pi_s = n_boxes[i] * pi_s
            mixture_lhs_s = lib_util.mm_pdf(mu_s, sig_s, pi_s, boxes_s, sum_comp=False)[0, :, 0]
            mixture_lhs_s = torch.max(mixture_lhs_s, dim=1)[0]
            mixture_nll_s = -torch.log(mixture_lhs_s + lib_util.epsilon)
            mog_nll_loss.append(mixture_nll_s)

    mog_nll_loss = torch.cat(mog_nll_loss, dim=0)
    return mog_nll_loss


def calc_cluster_nll(mu, sig, pi, boxes, n_boxes, coord_pdf=lib_util.gaussian_pdf):
    cluster_nll_loss = list()

    for i in range(mu.shape[0]):
        if n_boxes[i] == 0:
            pass
        else:
            pi_s = n_boxes[i] * pi[i:i + 1]
            mu_s, sig_s = mu[i:i + 1], sig[i:i + 1]
            boxes_s = boxes[i:i + 1, :n_boxes[i]]

            comp_lhs_s = lib_util.mm_pdf(
                mu_s, sig_s, pi_s, boxes_s, comp_pdf=coord_pdf, sum_comp=False)[0, :, 0]
            mixture_lhs_s = torch.sum(comp_lhs_s, dim=1)
            max_lhs_s = torch.max(comp_lhs_s, dim=1)[0]

            cluster_nll_s = -torch.log(max_lhs_s / (mixture_lhs_s + lib_util.epsilon) + lib_util.epsilon)
            cluster_nll_loss.append(cluster_nll_s)
            del cluster_nll_s

    cluster_nll_loss = torch.cat(cluster_nll_loss, dim=0)
    return cluster_nll_loss


def assign_cls_targets_single_class(pred_boxes, gt_boxes, gt_classes, matched_threshold=0.6, unmatched_threshold=0.45,
                          match_height=False, pos_fraction=None, sample_size=512):

    num_pred_boxes = pred_boxes.shape[0]
    num_gt = gt_boxes.shape[0]
    gt_classes = gt_classes.squeeze(-1)

    labels = torch.ones((num_pred_boxes,), dtype=torch.int64, device=pred_boxes.device) * -1
    gt_ids = torch.ones((num_pred_boxes,), dtype=torch.int64, device=pred_boxes.device) * -1

    if len(gt_boxes) > 0 and pred_boxes.shape[0] > 0:
        anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(pred_boxes[:, 0:7], gt_boxes[:, 0:7]) \
            if match_height else box_utils.boxes3d_nearest_bev_iou(pred_boxes[:, 0:7], gt_boxes[:, 0:7])
        anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda()
        anchor_to_gt_max = anchor_by_gt_overlap[
            torch.arange(num_pred_boxes, device=pred_boxes.device), anchor_to_gt_argmax
        ]
        gt_to_anchor_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=0)).cuda()
        gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=pred_boxes.device)]
        empty_gt_mask = gt_to_anchor_max == 0
        gt_to_anchor_max[empty_gt_mask] = -1

        pred_boxes_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0]
        gt_inds_force = anchor_to_gt_argmax[pred_boxes_with_max_overlap]
        labels[pred_boxes_with_max_overlap] = gt_classes[gt_inds_force]
        gt_ids[pred_boxes_with_max_overlap] = gt_inds_force#.int()

        pos_inds = anchor_to_gt_max >= matched_threshold
        gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
        labels[pos_inds] = gt_classes[gt_inds_over_thresh]
        gt_ids[pos_inds] = gt_inds_over_thresh#.int()
        bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0]
    else:
        bg_inds = torch.arange(num_pred_boxes, device=pred_boxes.device)

    fg_inds = (labels > 0).nonzero()[:, 0]

    if pos_fraction is not None:
        num_fg = int(pos_fraction * sample_size)
        if len(fg_inds) > num_fg:
            num_disabled = len(fg_inds) - num_fg
            disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
            labels[disable_inds] = -1
            fg_inds = (labels > 0).nonzero()[:, 0]

        num_bg = sample_size - (labels > 0).sum()
        if len(bg_inds) > num_bg:
            enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,))]
            labels[enable_inds] = 0
        # bg_inds = torch.nonzero(labels == 0)[:, 0]
    else:
        if len(gt_boxes) == 0 or pred_boxes.shape[0] == 0:
            labels[:] = 0
        else:
            labels[bg_inds] = 0
            labels[pred_boxes_with_max_overlap] = gt_classes[gt_inds_force]
    # print("* Car: {}, Ped: {}, Cyc: {}, Don't care: {}, total: {}".format((labels == 1).nonzero().shape[0],
    #                                                                     (labels == 2).nonzero().shape[0],
    #                                                                     (labels == 3).nonzero().shape[0],
    #                                                                     (labels == -1).nonzero().shape[0],
    #                                                                     labels.shape[0]))
    return labels

def assign_cls_targets(pred_boxes, gt_boxes, gt_classes,
                       matched_threshold=[0.6, 0.5, 0.5], unmatched_threshold=[0.45, 0.35, 0.35],
                          match_height=True, pos_fraction=None, sample_size=512):
    cls_targets = torch.ones((pred_boxes.shape[0],), dtype=torch.int32, device=pred_boxes.device) * -1
    bg_mask = torch.zeros((pred_boxes.shape[0],), dtype=torch.bool, device=pred_boxes.device)
    fg_mask = []
    dont_care_mask = []
    num_class = len(matched_threshold)
    for c in range(num_class):
        cur_class_mask = (gt_classes == c + 1)[:, 0]
        cur_boxes_s = gt_boxes[cur_class_mask]
        cur_labels_s = gt_classes[cur_class_mask]
        cls_target = assign_cls_targets_single_class(pred_boxes, cur_boxes_s, cur_labels_s,
                                 matched_threshold=matched_threshold[c], unmatched_threshold=unmatched_threshold[c],
                                 match_height=match_height, pos_fraction=pos_fraction, sample_size=sample_size)
        bg_mask[cls_target == 0] = True
        dont_care_mask.append(cls_target == -1)
        fg_mask.append(cls_target == (c+1))
    cls_targets[bg_mask] = 0
    for c in range(num_class):
        cls_targets[dont_care_mask[c]] = -1
        cls_targets[fg_mask[c]] = c+1
    # print("Car: {}, Ped: {}, Cyc: {}, Don't care: {}, total: {}".format((cls_targets == 1).nonzero().shape[0],
    #     (cls_targets == 2).nonzero().shape[0], (cls_targets == 3).nonzero().shape[0], (cls_targets == -1).nonzero().shape[0],
    #                                                                     cls_targets.shape[0]))
    return cls_targets


def get_cls_onehot_targets_weights(cls_preds, box_cls_labels, num_class):
    cared = box_cls_labels >= 0
    positives = box_cls_labels > 0
    negatives = box_cls_labels == 0
    negative_cls_weights = negatives * 1.0
    cls_weights = (negative_cls_weights + 1.0 * positives).float()
    if num_class == 1:
        # class agnostic
        box_cls_labels[positives] = 1

    pos_normalizer = positives.sum(0, keepdim=True).float()
    cls_weights /= torch.clamp(pos_normalizer, min=1.0)
    cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
    cls_targets = cls_targets.unsqueeze(dim=-1)

    cls_targets = cls_targets.squeeze(dim=-1)
    one_hot_targets = torch.zeros(
        *list(cls_targets.shape), num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
    )
    one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
    one_hot_targets = one_hot_targets[..., 1:]
    return one_hot_targets, cls_weights
