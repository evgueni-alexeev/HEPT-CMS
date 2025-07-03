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
    mask = (recons != 0) & (pts > pt_thres) #& (cluster_ids != -1)
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

def calculate_cluster_connectivity(cluster_indices, nearby_point_pairs):
    """
    Calculate what percentage of points in a cluster are connected to each other.
    Returns a float between 0 and 1 representing the connectivity percentage.
    """
    if len(cluster_indices) <= 1:
        return 0.0
        
    # Get all edges between points in this cluster
    cluster_edges = nearby_point_pairs[:, torch.isin(nearby_point_pairs[0], cluster_indices) & 
                                           torch.isin(nearby_point_pairs[1], cluster_indices)]
    
    # Initialize union-find data structures
    parent = {idx.item(): idx.item() for idx in cluster_indices}
    size = {idx.item(): 1 for idx in cluster_indices}
    
    def find_set(x):
        if parent[x] != x:
            parent[x] = find_set(parent[x])
        return parent[x]
    
    def union_set(a, b):
        ra = find_set(a)
        rb = find_set(b)
        if ra != rb:
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
    
    # Union all edges
    for i in range(cluster_edges.size(1)):
        u, v = cluster_edges[0, i].item(), cluster_edges[1, i].item()
        union_set(u, v)
    
    # Find the largest connected component
    component_sizes = {}
    for idx in cluster_indices:
        root = find_set(idx.item())
        component_sizes[root] = component_sizes.get(root, 0) + 1
    
    if not component_sizes:
        return 0.0
        
    largest_component_size = max(component_sizes.values())
    return largest_component_size / len(cluster_indices)

def save_cluster_connectivity(connectivity_metrics,parent_dir,event_id):
    event_id = int(event_id)
    metrics_file = parent_dir / f"event_{event_id}.txt"
    
    # Check if metrics include track lengths (3-tuple) or just connectivity (2-tuple)
    has_track_lengths = len(connectivity_metrics) > 0 and len(connectivity_metrics[0]) == 3
    
    if has_track_lengths:
        # Calculate statistics for this file with track lengths
        connectivities = [conn for _, conn, _ in connectivity_metrics]
        track_lengths = [tlen for _, _, tlen in connectivity_metrics]
        avg_connectivity = sum(connectivities) / len(connectivities) if connectivities else 0.0
        avg_track_length = sum(track_lengths) / len(track_lengths) if track_lengths else 0.0
        
        # Write detailed metrics with track lengths
        with open(metrics_file, "w") as f:
            f.write(f"Event {event_id} Statistics:\n")
            f.write(f"Number of clusters: {len(connectivity_metrics)}\n")
            f.write(f"Average connectivity: {avg_connectivity:.4f}\n")
            f.write(f"Average track length: {avg_track_length:.2f}\n")
            f.write(f"\nDetailed Cluster Metrics:\n")
            f.write(f"{'Cluster ID':<12} {'Connectivity':<12} {'Track Length':<12}\n")
            f.write("-" * 40 + "\n")
            for cluster_id, connectivity, track_length in connectivity_metrics:
                f.write(f"{cluster_id:<12} {connectivity:<12.4f} {track_length:<12}\n")
        
        # Update summary file with track lengths
        summary_file = parent_dir / "connectivity_summary.txt"
        with open(summary_file, "a") as f:
            f.write(f"{event_id}: connectivity={avg_connectivity:.4f}, track_len={avg_track_length:.2f} ({len(connectivity_metrics)} clusters)\n")
        
        # Update overall statistics file with track lengths
        stats_file = parent_dir / "connectivity_stats.txt"
        if not stats_file.exists():
            with open(stats_file, "w") as f:
                f.write("total_clusters,total_connectivity,weighted_avg_connectivity,total_track_length,avg_track_length\n")
        
        # Read current stats
        try:
            with open(stats_file, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:  # If we have existing stats
                    last_line = lines[-1].strip().split(",")
                    total_clusters = int(last_line[0])
                    total_connectivity = float(last_line[1])
                    total_track_length = float(last_line[3]) if len(last_line) > 3 else 0.0
                else:
                    total_clusters = 0
                    total_connectivity = 0.0
                    total_track_length = 0.0
        except (FileNotFoundError, IndexError, ValueError):
            total_clusters = 0
            total_connectivity = 0.0
            total_track_length = 0.0
        
        # Update stats
        total_clusters += len(connectivity_metrics)
        total_connectivity += sum(connectivities)
        total_track_length += sum(track_lengths)
        weighted_avg_connectivity = total_connectivity / total_clusters if total_clusters > 0 else 0.0
        avg_track_length_overall = total_track_length / total_clusters if total_clusters > 0 else 0.0
        
        # Write updated stats
        with open(stats_file, "w") as f:
            f.write("total_clusters,total_connectivity,weighted_avg_connectivity,total_track_length,avg_track_length\n")
            f.write(f"{total_clusters},{total_connectivity:.4f},{weighted_avg_connectivity:.4f},{total_track_length:.2f},{avg_track_length_overall:.2f}\n")
    
    else:
        # Original behavior for backward compatibility (2-tuple format)
        connectivities = [conn for _, conn in connectivity_metrics]
        avg_connectivity = sum(connectivities) / len(connectivities) if connectivities else 0.0
        
        # Write detailed metrics
        with open(metrics_file, "a") as f:
            f.write(f"File Statistics:\n")
            f.write(f"Number of clusters: {len(connectivity_metrics)}\n")
            f.write(f"Average connectivity: {avg_connectivity:.4f}\n")
            f.write(f"\nDetailed Cluster Metrics:\n")
            for cluster_id, connectivity in connectivity_metrics:
                f.write(f"Cluster {cluster_id}: {connectivity:.4f}\n")
        
        # Update summary file
        summary_file = parent_dir / "connectivity_summary.txt"
        with open(summary_file, "a") as f:
            f.write(f"{event_id}: {avg_connectivity:.4f} ({len(connectivity_metrics)} clusters)\n")
        
        # Update overall statistics file
        stats_file = parent_dir / "connectivity_stats.txt"
        if not stats_file.exists():
            with open(stats_file, "w") as f:
                f.write("total_clusters,total_connectivity,weighted_avg\n")
        
        # Read current stats
        try:
            with open(stats_file, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:  # If we have existing stats
                    last_line = lines[-1].strip().split(",")
                    total_clusters = int(last_line[0])
                    total_connectivity = float(last_line[1])
                else:
                    total_clusters = 0
                    total_connectivity = 0.0
        except FileNotFoundError:
            total_clusters = 0
            total_connectivity = 0.0
        
        # Update stats
        total_clusters += len(connectivity_metrics)
        total_connectivity += sum(connectivities)
        weighted_avg = total_connectivity / total_clusters if total_clusters > 0 else 0.0
        
        # Write updated stats
        with open(stats_file, "w") as f:
            f.write("total_clusters,total_connectivity,weighted_avg\n")
            f.write(f"{total_clusters},{total_connectivity:.4f},{weighted_avg:.4f}\n")
    
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