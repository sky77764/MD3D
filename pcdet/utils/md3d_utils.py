import os
import math
import torch
import numpy as np
from torchvision.ops import box_iou

epsilon = 1e-12
float_epsilon = 1e-36
double_epsilon = 1e-300
float_max = 1e+36


def cvt_torch2numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        if tensor.is_cuda:
            tensor = tensor.detach().cpu().numpy()
        else:
            tensor = tensor.detach().numpy()
    elif isinstance(tensor, list) or isinstance(tensor, tuple):
        for i in range(len(tensor)):
            tensor[i] = cvt_torch2numpy(tensor[i])
    elif isinstance(tensor, dict):
        for key in tensor.keys():
            tensor[key] = cvt_torch2numpy(tensor[key])
    return tensor


def gaussian_pdf(x, mu, sig):
    # make |mu|=K copies of y, subtract mu, divide by sigma
    # print(x.shape, mu.shape, sig.shape)
    dist = ((x - mu) / sig) ** 2
    # result = (x - mu) / sig

    result = -0.5 * dist
    result = torch.exp(result) / (sig * math.sqrt(2.0 * math.pi))
    return result

def multivariate_gaussian_pdf(x, mu, U, sqrt_inv_det_cov, get_log_prob=False):
    """
    x                       [B, N, 7, 1]
    mu                      [B, 1, 7, K]
    U                       [B, 7, 7, K]
    sqrt_inv_det_cov        [B, 1, K]
    """
    D = x.shape[2]
    diff = (x - mu).transpose(1, 3)   # [B, K, 7, N]
    U = U.transpose(2, 3).transpose(1, 2)   # [B, K, 7, 7]
    z = torch.matmul(U, diff)   # [B, K, 7, N]

    result = -0.5 * (z ** 2)
    if get_log_prob:
        result = result.sum(dim=-2, keepdim=True) + torch.log(sqrt_inv_det_cov.transpose(1, 2).unsqueeze(-1)) - (D/2.0)*math.log(2*math.pi)
    else:
        result = torch.exp(result).prod(dim=-2, keepdim=True) * sqrt_inv_det_cov.transpose(1, 2).unsqueeze(-1) / ((2.0*math.pi)**(D/2.0))
    result = result.transpose(1, 3)

    return result


def cauchy_pdf(x, loc, sc):
    # x:    (batch, #x, 7, 1)
    # loc:  (batch, 1, 7, #comp)
    # sc:   (batch, 1, 7, #comp)

    dist = ((x - loc) / sc) ** 2

    # dist: (batch, #x, 7, #comp)
    result = 1 / (math.pi * sc * (dist + 1))
    # print("result.min(): ", result.min().item(), ", result.max(): ", result.max().item())
    return result

def laplace_pdf(x, mu, b):
    result = 1.0 / (2.0 * b) * torch.exp(-torch.abs(x-mu) / b)
    return result


def mm_pdf(mu, sig, pi, points, sum_comp=True, comp_pdf=gaussian_pdf, U=None, sqrt_inv_det_cov=None, get_log_prob=False):

    mu, pi = mu.unsqueeze(dim=1),  pi.unsqueeze(dim=1)
    points = points.unsqueeze(dim=3)
    if U is None:
        # mu, sig shape:    (batch, 1, 7, n_gauss)
        # pi shape:         (batch, 1, 1, n_gauss)
        # points shape:     (batch, n_points, 7, 1)
        sig = sig.unsqueeze(dim=1)
        result = comp_pdf(points, mu, sig)

        # result.shape:     (batch, n_points, 7, n_gauss)
        result = torch.prod(result, dim=2, keepdim=True)
        result = pi * result

    else:   # Multivariate
        split_indices = [U_split.shape[1] for U_split in U]
        points_split = list(torch.split(points, split_indices, dim=2))
        mu_split = list(torch.split(mu.clone(), split_indices, dim=2))

        result = 0 if get_log_prob else 1
        for idx in range(len(U)):
            result_split = comp_pdf(points_split[idx], mu_split[idx], U[idx], sqrt_inv_det_cov[idx], get_log_prob=get_log_prob)
            if get_log_prob:
                result_split = torch.sum(result_split, dim=2, keepdim=True)
                result = result + result_split
            else:
                result_split = torch.prod(result_split, dim=2, keepdim=True)
                result = result * result_split
        if get_log_prob:
            result = torch.log(pi) + result
        else:
            result = pi * result

    if sum_comp:
        result = torch.sum(result, dim=3)
        # result.shape: (batch, n_points, 1)
    return result


def mm_pdf_s(mu_s, sig_s, pi_s, points_s, sum_comp=True, comp_pdf=gaussian_pdf):
    mu, sig, pi = mu_s.unsqueeze(dim=0), sig_s.unsqueeze(dim=0), pi_s.unsqueeze(dim=0)
    points_s = points_s.unsqueeze(dim=2)
    # mu, sig shape:    (1, 7, n_gauss)
    # pi shape:         (1, 1, n_gauss)
    # points shape:     (n_points, 7, 1)

    result = comp_pdf(points_s, mu_s, sig_s)
    # result.shape:     (n_points, 7, n_gauss)
    result = torch.prod(result, dim=1, keepdim=True)
    # result.shape:     (n_points, 1, n_gauss)
    if pi is not None:
        result = (pi * result)[:, 0]
        # result.shape: (n_points, n_gauss)
    if sum_comp:
        result = torch.sum(result, dim=1)
        # result.shape: (n_points)
    return result


def category_pmf(clsprobs, onehot_labels):
    # clsprobs: (batch, #classes, #comp)
    # labels:   (batch, #samples, #classes)
    clsprobs = clsprobs.unsqueeze(dim=1)
    onehot_labels = onehot_labels.unsqueeze(dim=3)
    cat_probs = torch.prod(clsprobs ** onehot_labels, dim=2, keepdim=True)
    return cat_probs


def category_pmf_s(clsprobs, onehot_labels):
    # clsprobs: (#classes, #comp)
    # labels:   (#samples, #classes)
    clsprobs = clsprobs.unsqueeze(dim=0)
    onehot_labels = onehot_labels.unsqueeze(dim=2)
    cat_probs = torch.prod(clsprobs ** onehot_labels, dim=1)
    return cat_probs


def bernoulli_pmf_s(probs, labels):
    probs = probs.unsqueeze(dim=0)
    labels = labels.unsqueeze(dim=2)
    # probs:    (1, #classes, #comp)
    # labels:   (#samples, #classes, 1)
    ber_probs = (labels * probs) + ((1 - labels) * (1 - probs))
    ber_probs = torch.prod(ber_probs, dim=1)
    return ber_probs


def create_coord_map(coord_map_size, coord_range):
    # gauss_w: 4 --> ((0, 1, 2, 3), ...)
    y_map = np.array(list(range(coord_map_size[0])) * coord_map_size[1]).astype(np.float32)
    x_map = np.array(list(range(coord_map_size[1])) * coord_map_size[0]).astype(np.float32)

    y_map = y_map.reshape((1, 1, coord_map_size[1], coord_map_size[0]))
    y_map = y_map.transpose((0, 1, 3, 2))
    x_map = x_map.reshape((1, 1, coord_map_size[0], coord_map_size[1]))

    unit_intv_y = coord_range[0] / coord_map_size[0]
    unit_intv_x = coord_range[1] / coord_map_size[1]

    x_map = x_map * unit_intv_x + unit_intv_x / 2
    y_map = y_map * unit_intv_y + unit_intv_y / 2
    return np.concatenate((x_map, y_map), axis=1)

def create_coord_map_lidar(coord_map_size, coord_range, xy_lidar_to_bev, point_cloud_range):
    # gauss_w: 4 --> ((0, 1, 2, 3), ...)
    xy_map = create_coord_map(coord_map_size, coord_range)
    xy_map[0, 0] = xy_map[0, 0] / xy_lidar_to_bev[0] + point_cloud_range[0]
    xy_map[0, 1] = xy_map[0, 1] / xy_lidar_to_bev[1] + point_cloud_range[1]

    return xy_map


def sample_boxes_s(pi_s, mu_s, n_samples):
    # pi_s: (1, #comp)
    # mu_s: (4, #comp)
    sampler = torch.distributions.categorical.Categorical(pi_s[0])
    sampled_indices = sampler.sample((n_samples,))
    sampled_boxes_s = mu_s[:, sampled_indices]
    return sampled_boxes_s
