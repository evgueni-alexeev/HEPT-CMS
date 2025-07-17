import math
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from numba import jit
from torch.optim import Adam, AdamW, RMSprop, Adamax, LBFGS, NAdam, RAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR, StepLR

from losses import InfoNCELoss, FocalLoss

def point_filter(cluster_ids, recons, pts, pt_thres):
    mask = (recons != 0) & (pts > pt_thres) & (cluster_ids >= 0)
    return mask

@torch.no_grad()
def calc_AP_at_k(embeddings, cluster_ids, track_lengths, mask, dist_metric, batch_size=None):

    cluster_ids = cluster_ids.cpu().numpy()
    track_lengths = track_lengths.cpu().numpy()
    mask = mask.cpu().numpy()

    num_points = embeddings.shape[0]
    if batch_size is None:
        batch_size = num_points

    precision_at_k = []

    for start_index in range(0, num_points, batch_size):
        end_index = min(start_index + batch_size, num_points)

        batch_mask = mask[start_index:end_index]
        batch_embeddings = embeddings[start_index:end_index][batch_mask]
        batch_cluster_ids = cluster_ids[start_index:end_index][batch_mask]
        batch_tracklens = track_lengths[start_index:end_index][batch_mask]

        # Compute pairwise distances from the batch points to ALL points
        if "l2" in dist_metric:
            dist_mat_batch = torch.cdist(batch_embeddings, embeddings, p=2.0)
        elif dist_metric == "cosine":
            dist_mat_batch = 1 - F.cosine_similarity(batch_embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        else:
            raise NotImplementedError

        # True neighbour count per query: track_length - 1 (excluding itself)
        k_list = (batch_tracklens - 1).astype(np.int64)
        K = int(k_list.max())

        # indices of K nearest neighbours (plus self)
        indices = dist_mat_batch.topk(K + 1, dim=1, largest=False, sorted=True)[1].cpu().numpy()

        AP = calc_scores(K, k_list, indices, cluster_ids, batch_cluster_ids)
        precision_at_k.extend(AP)

    mean_ap = float(np.mean(precision_at_k)) if precision_at_k else 0.0
    return mean_ap

@jit(nopython=True)
def calc_scores(K, k_list, indices, cluster_ids, batch_cluster_ids):
    prec = []
    for i, k in enumerate(k_list):
        if k == 0:
            continue

        # slice the k nearest neighbors
        neighbors = indices[i, 1 : K + 1]

        # Retrieve the labels of the k nearest neighbors
        neighbor_labels = cluster_ids[neighbors]

        # check if neighbor labels match the expanded labels (precision)
        matches = neighbor_labels == batch_cluster_ids[i]

        precision_at_k = matches[:k].sum() / k

        prec.append(precision_at_k)

    return prec
    
def get_loss(loss_name, loss_kwargs):
    if loss_name == "infonce":
        return InfoNCELoss(**loss_kwargs)
    elif loss_name == "crossentropy":
        return torch.nn.BCEWithLogitsLoss()
    elif loss_name == "focal":
        return FocalLoss(**loss_kwargs)
    else:
        raise NotImplementedError

def get_optimizer(parameters, optimizer_name, optimizer_kwargs):
    if optimizer_name.lower() == "adam":
        return Adam(parameters, **optimizer_kwargs)
    elif optimizer_name.lower() == "adamw":
        return AdamW(parameters, **optimizer_kwargs)
    elif optimizer_name.lower() == "rmsprop":
        return RMSprop(parameters, **optimizer_kwargs)
    elif optimizer_name.lower() == "adamax":
        return Adamax(parameters, **optimizer_kwargs)
    elif optimizer_name.lower() == "lbfgs":
        return LBFGS(parameters, **optimizer_kwargs)
    elif optimizer_name.lower() == "nadam":
        return NAdam(parameters, **optimizer_kwargs)
    elif optimizer_name.lower() == "radam":
        return RAdam(parameters, **optimizer_kwargs)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported!")

def get_lr_scheduler(optimizer, lr_scheduler_name, lr_scheduler_kwargs):
    if lr_scheduler_name is None:
        return None
    elif lr_scheduler_name == "impatient":
        lr_scheduler_kwargs.pop("num_training_steps", None)
        return ReduceLROnPlateau(optimizer, **lr_scheduler_kwargs)
    elif lr_scheduler_name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, **lr_scheduler_kwargs)
    elif lr_scheduler_name == "step":
        lr_scheduler_kwargs.pop("num_training_steps", None)
        return StepLR(optimizer, **lr_scheduler_kwargs)
    else:
        raise ValueError(f"LR scheduler {lr_scheduler_name} not supported!")

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, eta_min, num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)