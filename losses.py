import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import segment_csr

class InfoNCELoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.tau = kwargs["tau"]
        self.dist_metric = kwargs["dist_metric"]
        self.sigma = kwargs["sigma"]

    def forward(self, x, point_pairs, cluster_ids, recons, pts, tracklen=None, **kwargs):
        all_pos_pair_mask = cluster_ids[point_pairs[0]] == cluster_ids[point_pairs[1]]
        non_zero_pid_point_pairs = (cluster_ids[point_pairs[0]] != -1) & (cluster_ids[point_pairs[1]] != -1)
        all_pos_pair_mask = all_pos_pair_mask & non_zero_pid_point_pairs
        all_neg_pair_mask = ~all_pos_pair_mask

        if self.dist_metric == "cosine":
            similarity = F.cosine_similarity(x[point_pairs[0]], x[point_pairs[1]], dim=-1)
        elif self.dist_metric == "l2_rbf":
            l2_dist = torch.linalg.norm(x[point_pairs[0]] - x[point_pairs[1]], ord=2, dim=-1)
            similarity = torch.exp(-l2_dist / (2 * self.sigma**2))
        elif self.dist_metric == "l2_inverse":
            l2_dist = torch.linalg.norm(x[point_pairs[0]] - x[point_pairs[1]], ord=2, dim=-1)
            similarity = 1.0 / (l2_dist + 1.0)
        else:
            raise NotImplementedError

        loss_per_pos_pair = self.calc_info_nce(x, similarity, point_pairs, all_pos_pair_mask, all_neg_pair_mask, tracklen)

        return torch.mean(loss_per_pos_pair)

    def calc_info_nce(self, x, similarity, all_pairs, all_pos_pair_mask, all_neg_pair_mask, tracklen=None):
        max_sim = (similarity / self.tau).max()
        exp_sim = torch.exp(similarity / self.tau - max_sim)

        pos_exp_sim = exp_sim[all_pos_pair_mask]
        neg_exp_sim = exp_sim[all_neg_pair_mask]

        # Model has harder time reconstructing shorter tracks, so we downweight negatives by relative track length
        if tracklen is not None:
            max_tracklen = tracklen.max().float()
            if max_tracklen > 0:
                anchor_tracklen = tracklen[all_pairs[0][all_neg_pair_mask]]
                tracklen_weights = anchor_tracklen.float() / max_tracklen
                neg_exp_sim = neg_exp_sim * tracklen_weights

        numerator = pos_exp_sim
        neg_indices = all_pairs[0][all_neg_pair_mask]
        range_size = len(x)

        denominator = deterministic_scatter(neg_exp_sim, neg_indices, range_size, reduce="sum").clamp(min=0)

        denominator = denominator[all_pairs[0][all_pos_pair_mask]]
        loss_per_pos_pair = -torch.log(numerator / (numerator + denominator))
        return loss_per_pos_pair

def deterministic_scatter(src, index, range_size, reduce="sum"):
    device, idx_dtype = index.device, index.dtype

    dummy_idx = torch.tensor([range_size - 1], device=device, dtype=idx_dtype)
    dummy_src = torch.zeros((1,), device=src.device, dtype=src.dtype)

    index = torch.cat([index, dummy_idx], dim=0)
    src = torch.cat([src, dummy_src], dim=0)

    sorted_arg = torch.argsort(index)
    sorted_index = index[sorted_arg]
    sorted_src = src[sorted_arg]

    unique_groups, counts = torch.unique_consecutive(sorted_index, return_counts=True)
    indptr = torch.zeros(len(unique_groups) + 1, device=device, dtype=torch.long)
    indptr[1:] = torch.cumsum(counts, dim=0)

    out = segment_csr(sorted_src, indptr, reduce=reduce)

    if out.size(0) < range_size:
        pad_len = range_size - out.size(0)
        out = torch.cat([out, out.new_zeros(pad_len)], dim=0)

    return out

class FocalLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        alpha = 1.0/(kwargs["imbalance"]+1)
        assert alpha <= 1.0
        self.dynamic_alpha = kwargs["use_dynamic_alpha"]
        self.alpha = torch.tensor([alpha,1.0-alpha])
        self.gamma = kwargs["gamma"]
        self.imb = None
        self.multiplier = kwargs["multiplier"] if kwargs["multiplier"] is not None else 1.0

    def forward(self, inputs, targets, imbalance_ratio=None):
        self.imb = self.multiplier*imbalance_ratio
        if self.dynamic_alpha:
            self.alpha = torch.tensor([1.0/(self.imb+1.0), self.imb/(self.imb+1.0)])
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        targets = targets.type(torch.long)
        at = self.alpha.to(targets.device).gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*((1 - pt)**self.gamma)*BCE_loss
        return F_loss.mean()
