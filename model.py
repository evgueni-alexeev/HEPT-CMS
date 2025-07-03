"""
Transformer model with HEPT attention using E2LSH.
Model based on https://arxiv.org/pdf/2402.12535
Attention code adapted from https://github.com/Graph-COM/HEPT

"""

import torch
from torch import nn
from torch_geometric.nn import MLP
from torch.utils.checkpoint import checkpoint
from einops import rearrange
import math


#============================================
#          MAIN TRANSFORMER STRUCTURE
#============================================

class Transformer(nn.Module):
    def __init__(self, in_dim, coords_dim, task, **kwargs):
        super().__init__()
        self.task = task
        self.n_layers = kwargs["n_layers"]
        self.h_dim = kwargs["h_dim"]

        self.LSH_DIMS = kwargs.get("lsh_dims", 3)
        assert self.LSH_DIMS in [1, 2, 3], f"LSH_DIMS must be 1, 2, or 3, got {self.LSH_DIMS}"

        self.use_ckpt = kwargs.get("use_ckpt", False)

        # discrete detector layers (Anchor MD: 1,...,10 vals, End MD: 2,...,11 vals possible) to embedding
        self.layer0_emb = nn.Embedding(11, 1)
        self.layer1_emb = nn.Embedding(12, 1)

        # input feature encoder
        self.feat_encoder = nn.Sequential(
            nn.Linear(in_dim, self.h_dim),
            # nn.LayerNorm(self.h_dim), -- input data (mostly) standardized to 0 mean, unit variance in dataset step
            nn.SiLU(),
            nn.Linear(self.h_dim, self.h_dim),
        )

        # HEPT self-attention blocks
        self.attns = nn.ModuleList()
        for _ in range(self.n_layers):
            self.attns.append(Attn(coords_dim, **kwargs))

        # MLP layer settings
        self.in_MLP = round(kwargs["W_out_hdim_factor"]*self.h_dim) # original: int(self.h_dim // 2) = 12
        self.n_MLP = kwargs["h_dim_MLP"]
        self.d_MLP = kwargs["n_layers_MLP"]
        self.act_MLP = kwargs["act_MLP"]

        self.dropout = nn.Dropout(kwargs["dropout"])
        self.W = nn.Linear(self.h_dim * (self.n_layers + 1), self.in_MLP , bias=False)

        self.norm_mid = nn.LayerNorm(self.in_MLP)

        self.mlp_out = MLP(
            in_channels=self.in_MLP ,
            out_channels=self.in_MLP ,
            hidden_channels=self.n_MLP,
            num_layers=self.d_MLP,
            norm="layer_norm",
            act=self.act_MLP,
            norm_kwargs={"mode": "node"},
        )

        self.helper_params = {}

        self.helper_params["block_size"] = kwargs["block_size"]

        # regions for LSH: n_hashes → num OR hashes, LSH_DIMS = num AND hashes (top 1,2 or 3 LDA coords)
        self.register_buffer("regions", get_regions(kwargs["num_regions"], kwargs["n_hashes"], kwargs["num_heads"], num_and_hashes=self.LSH_DIMS), persistent=True)
        self.helper_params["regions"] = self.regions
        self.helper_params["num_heads"] = kwargs["num_heads"]

        # final output projection if task is pileup
        if self.task == "pileup":
            self.out_proj = nn.Linear(self.in_MLP, 1)
        
        # -------------------------------------------------
        #               Weight initializations
        # -------------------------------------------------
        # Linear → SiLU :  w ~ N(0, 2.952 / d_in),  b ~ N(0, 0.04)                  ref: https://arxiv.org/abs/1805.08266
        # Linear → ReLU :  He/normal (fan_in mode, nonlinearity='relu'), b = 0

        def _init_weights(mod):
            # Embedding separately
            if isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, mean=0.0, std=1.0)
                return

            # Treat nn.Linear and PyG Linear (or any module with 2-D weight & bias attr) as linear-like.
            is_linear_like = (
                isinstance(mod, nn.Linear) or (
                    hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor) and mod.weight.dim() == 2 and hasattr(mod, "bias")
                )
            )
            # Check all linear layers whether they use SiLU or ReLU, and initialize weights accordingly
            if is_linear_like:
                if getattr(mod, "_uses_silu", False):
                    fan_in = mod.weight.size(1)
                    std = math.sqrt(2.952 / fan_in)
                    nn.init.normal_(mod.weight, mean=0.0, std=std)
                    if mod.bias is not None:
                        nn.init.normal_(mod.bias, mean=0.0, std=0.2)  # var = 0.04
                else:
                    nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.LayerNorm):
                nn.init.ones_(mod.weight)
                nn.init.zeros_(mod.bias)

        # Currently all linear layers HARD-CODED for SiLU

        def _mark_silu_linear(mod):
            """Helper: flag linear-like modules for SiLU weight initialisation."""
            if hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor) and mod.weight.dim() == 2:
                mod._uses_silu = True

        # 1) MLP head
        for sub_mod in self.mlp_out.modules():
            _mark_silu_linear(sub_mod)

        # 2) Input feature encoder
        for sub_mod in self.feat_encoder.modules():
            _mark_silu_linear(sub_mod)

        # 3) Attention blocks (projections + feed-forward nets)
        for attn_block in self.attns:
            _mark_silu_linear(attn_block.w_q)
            _mark_silu_linear(attn_block.w_k)
            _mark_silu_linear(attn_block.w_v)

            for sub_mod in attn_block.ff.modules():
                _mark_silu_linear(sub_mod)
        
        # Apply the initialisation to the whole sub-tree
        self.apply(_init_weights)

    def forward(self, data):
        """
        3D LSH Transformer forward pass.
        
        Args:
            data: PyTorch Geometric Data object containing:
                x: Tensor of shape (N, in_dim), where N is the total number of points in the batch
                coords: Tensor of shape (N, coords_dim), where coords_dim >= 3
                    - coords[:, 0]: 1st LDA coordinate, mostly eta + dz (for LSH partitioning)
                    - coords[:, 1]: 2nd LDA coordinate, mostly cos(phi), dx (for LSH partitioning) 
                    - coords[:, 2]: 3rd LDA coordinate, mostly sin(phi), dy (for LSH partitioning)
                    - coords[:, 3:]: additional coordinates (used for attention but not LSH)
                batch: Tensor of shape (N,)            
        """
        if isinstance(data, dict):
            x, coords, batch, self.use_ckpt = data["x"], data["coords"], data["batch"], False
        else:
            x, coords, batch = data.x, data.coords, data.batch

        # swap out last two columns of x (start and end layer indices) with embeddings
        MD_A_layer = self.layer0_emb(x[..., -2].long())
        MD_E_layer = self.layer1_emb(x[..., -1].long())
        x = torch.cat((x[..., :-2], MD_A_layer, MD_E_layer), dim=-1)
        
        x, kwargs, unpad_seq = prepare_input(x, coords, batch, self.helper_params, lsh_dims = self.LSH_DIMS)

        encoded_x = self.feat_encoder(x)
        all_encoded_x = [encoded_x]
        for i in range(self.n_layers):
            if self.use_ckpt:
                encoded_x = checkpoint(self.attns[i], encoded_x, kwargs)
            else:
                encoded_x = self.attns[i](encoded_x, kwargs)
            all_encoded_x.append(encoded_x)

        encoded_x = self.norm_mid(self.W(torch.cat(all_encoded_x, dim=-1)))
        out = encoded_x + self.dropout(self.mlp_out(encoded_x))

        if self.task == "pileup":
            out = self.out_proj(out)        # using binary_cross_entropy_with_logits in losses -- no need to sigmoid

        return out[unpad_seq]



#============================================
#              HEPT ATTENTION
#============================================

class Attn(nn.Module):
    def __init__(self, coords_dim, **kwargs):
        super().__init__()
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]

        self.w_q = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_k = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)
        self.w_v = nn.Linear(self.dim_per_head, self.dim_per_head * self.num_heads, bias=False)

        # +3 for data.pos
        self.attn = HEPTAttention(self.dim_per_head + coords_dim, **kwargs)

        self.dropout = nn.Dropout(kwargs["dropout"])
        self.norm1 = nn.LayerNorm(self.dim_per_head)
        self.norm2 = nn.LayerNorm(self.dim_per_head)
        self.ff = nn.Sequential(
            nn.Linear(self.dim_per_head, self.dim_per_head),
            nn.SiLU(),
            nn.Linear(self.dim_per_head, self.dim_per_head),
        )

        # independent learned weights to scale each coordinate in attention kernel (original used dR so needed (coords_dim - 1) in order for eta and phi to use same weight)
        self.w_rpe = nn.Linear(kwargs["num_w_per_dist"] * coords_dim, self.num_heads * self.dim_per_head)

    def forward(self, x, kwargs):
        x_normed = self.norm1(x)
        q, k, v = self.w_q(x_normed), self.w_k(x_normed), self.w_v(x_normed)
        aggr_out = self.attn(q, k, v, pe=kwargs["coords"], w_rpe=self.w_rpe, **kwargs)

        x = x + self.dropout(aggr_out)
        ff_output = self.ff(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x

class HEPTAttention(nn.Module):
    def __init__(self, hash_dim, **kwargs):
        super().__init__()
        self.dim_per_head = kwargs["h_dim"]
        self.num_heads = kwargs["num_heads"]
        self.out_linear = nn.Linear(self.num_heads * self.dim_per_head, self.dim_per_head)

        self.block_size = kwargs["block_size"]
        self.n_hashes = kwargs["n_hashes"]
        self.num_w_per_dist = kwargs["num_w_per_dist"]
        self.e2lsh = E2LSH(n_hashes=self.n_hashes, n_heads=self.num_heads, dim=hash_dim)

    def forward(self, query, key, value, **kwargs):
        query = query.view(-1, self.num_heads, self.dim_per_head)
        key = key.view(-1, self.num_heads, self.dim_per_head)
        value = value.view(-1, self.num_heads, self.dim_per_head)

        w = rearrange(
            kwargs["w_rpe"].weight,
            "(h d) (r k) -> h d r k",
            h=self.num_heads,
            d=self.dim_per_head,
            k=self.num_w_per_dist,
        )
        q_hat, k_hat = prep_qk(query, key, w, kwargs["coords"])

        q_hat = rearrange(q_hat, "n h d -> h n d")
        k_hat = rearrange(k_hat, "n h d -> h n d")
        value = rearrange(value, "n h d -> h n d")

        q_hashed, k_hashed, hash_shift = lsh_mapping(self.e2lsh, q_hat, k_hat)

        combined_shifts = kwargs["combined_shifts"] * hash_shift
        q_hashed = q_hashed + combined_shifts
        k_hashed = k_hashed + combined_shifts

        q_positions = q_hashed.argsort(dim=-1)
        k_positions = k_hashed.argsort(dim=-1)

        s_query = sort_to_buckets(q_hat, q_positions, self.block_size)
        s_key = sort_to_buckets(k_hat, k_positions, self.block_size)
        s_value = sort_to_buckets(value, k_positions, self.block_size)

        denom, so = qkv_res(s_query, s_key, s_value)

        q_rev_positions = invert_permutation(q_positions)
        o = unsort_from_buckets(so, q_rev_positions)
        logits = unsort_from_buckets(denom, q_rev_positions)
        out = o.sum(dim=0) / logits.sum(dim=0)
        out = self.out_linear(rearrange(out, "h n d -> n (h d)"))
        return out



#============================================
#             HELPER FUNCTIONS
#============================================

def bit_shift(base, shift_idx):
    max_base = base.max(dim=1, keepdim=True).values
    num_bits = torch.ceil(torch.log2(max_base + 1)).long()
    return (shift_idx << num_bits) | base

def pad_and_unpad(batch, block_size, region_indices, raw_sizes):
    padded_sizes = ((raw_sizes + block_size - 1) // block_size) * block_size
    pad_sizes = padded_sizes - raw_sizes

    pad_cumsum = padded_sizes.cumsum(0)
    pad_seq = torch.arange(pad_cumsum[-1], device=batch.device)
    unpad_seq = torch.ones(pad_cumsum[-1], device=batch.device).bool()

    sorted_region_indices = region_indices.argsort()
    for i in range(len(raw_sizes)):
        idx_to_fill = pad_cumsum[i] - block_size - pad_sizes[i] + torch.arange(pad_sizes[i], device=batch.device)
        if i >= 1:
            pad_seq[pad_cumsum[i - 1] :] -= pad_sizes[i - 1]
            idx_to_fill -= pad_sizes[:i].sum()
        pad_seq[pad_cumsum[i] - pad_sizes[i] : pad_cumsum[i]] = sorted_region_indices[idx_to_fill]
        unpad_seq[pad_cumsum[i] - pad_sizes[i] : pad_cumsum[i]] = False
    return pad_seq, unpad_seq

def prepare_input(x, coords, batch, helper_params,lsh_dims=3):
    kwargs = {}
    regions = rearrange(helper_params["regions"], "c a h -> a (c h)")
    with torch.no_grad():
        block_size, num_heads = helper_params["block_size"], helper_params["num_heads"]
        graph_sizes = batch.bincount()
        graph_size_cumsum = graph_sizes.cumsum(0)

        # 3D LSH: use data.coords, which are top 3 LDA coordinates by default
        region_1, region_2, region_3 = [], [], []
        for graph_idx in range(len(graph_size_cumsum)):
            start_idx = 0 if graph_idx == 0 else graph_size_cumsum[graph_idx - 1]
            end_idx = graph_size_cumsum[graph_idx]
            sorted_eta_idx = torch.argsort(coords[start_idx:end_idx, 0], dim=-1)
            sorted_phi_idx = torch.argsort(coords[start_idx:end_idx, 1], dim=-1)
            sorted_third_idx = torch.argsort(coords[start_idx:end_idx, 2], dim=-1)

            region_1.append(quantile_partition(sorted_eta_idx, regions[0][:, None]))
            if lsh_dims >= 2:
                region_2.append(quantile_partition(sorted_phi_idx, regions[1][:, None]))
                if lsh_dims == 3:
                    region_3.append(quantile_partition(sorted_third_idx, regions[2][:, None]))
        
        region_1 = torch.cat(region_1, dim=-1)
        if lsh_dims >= 2:
            region_2 = torch.cat(region_2, dim=-1)
            if lsh_dims == 3:
                region_3 = torch.cat(region_3, dim=-1)

        # Combine 1, 2, or 3 dimensions using bit shifting
        if lsh_dims == 1:
            combined_shifts = bit_shift(region_1.long(),batch[None])
        else:
            combined_shifts = bit_shift(region_1.long(), region_2.long())
            if lsh_dims == 3:
                combined_shifts = bit_shift(combined_shifts, region_3.long())
            combined_shifts = bit_shift(combined_shifts, batch[None])
        combined_shifts = rearrange(combined_shifts, "(c h) n -> c h n", h=num_heads)

        pad_seq, unpad_seq = pad_and_unpad(batch, block_size, combined_shifts[0, 0], graph_sizes)
        x = x[pad_seq]
        kwargs["combined_shifts"] = combined_shifts[..., pad_seq]
        kwargs["coords"] = coords[pad_seq]
    return x, kwargs, unpad_seq

def sort_to_buckets(x, perm, bucketsz):
    return rearrange(
        batched_index_select(rearrange(x, "b s d -> 1 b s d"), perm),
        "h b (nbuckets bucketsz) d -> h b nbuckets bucketsz d",
        bucketsz=bucketsz,
    )

def unsort_from_buckets(s_x, perm_inverse):
    b_x = rearrange(s_x, "h b nbuckets bucketsz d -> h b (nbuckets bucketsz) d")
    return batched_index_select(b_x, perm_inverse)

def qkv_res(s_query, s_key, s_value):
    # Gaussian attention kernel, exp(-1/2|q-k|^2) = exp(q·k - 1/2(|q|^2 + |k|^2))
    q_sq_05 = -0.5 * (s_query**2).sum(dim=-1, keepdim=True)
    k_sq_05 = -0.5 * (s_key**2).sum(dim=-1, keepdim=True)

    clustered_dists = torch.einsum("...id,...jd->...ij", s_query, s_key)
    clustered_dists = (clustered_dists + q_sq_05 + k_sq_05.transpose(-1, -2)).clamp(max=0.0).exp()

    denom = clustered_dists.sum(dim=-1, keepdim=True) + (1e-20)
    qk = clustered_dists

    so = torch.einsum("...ij,...jd->...id", qk, s_value)
    return denom, so

def prep_qk(query, key, w, coords):
    # collapse feature & kernel dims to obtain one positive weight per (head, coord)
    qw = w.sum(dim=1).clamp(max=50).exp().sum(dim=-1)
    # qw = torch.cat([qw[:, :1], qw], dim=-1) -- original HEPT implementation used dR, so duplicated weight for eta and phi

    sqrt_w_r = torch.sqrt(2 * qw)[None] * coords[:, None]
    q_hat = torch.cat([query, sqrt_w_r], dim=-1)
    k_hat = torch.cat([key, sqrt_w_r], dim=-1)

    return q_hat, k_hat

def quantile_partition(sorted_indices, num_regions):
    total_elements = sorted_indices.shape[-1]
    region_size = torch.ceil(total_elements / num_regions)
    inverse_indices = torch.argsort(sorted_indices, dim=-1)

    base = torch.arange(total_elements, device=sorted_indices.device)[None]
    region_size = region_size.to(base.device)
    region_indices = base // region_size + 1
    reassigned_regions = region_indices[:, inverse_indices]
    return reassigned_regions

def get_regions(num_regions, num_or_hashes, num_heads, num_and_hashes):
    lb = 2
    ub = 2 * num_regions ** (1 / num_and_hashes) - lb
    regions = []
    for _ in range(num_or_hashes * num_heads):
        region = []
        for _ in range(num_and_hashes):
            a = torch.rand(1).item() * (ub - lb) + lb
            region.append(a)
        regions.append(region)
    regions = torch.tensor(regions)
    regions = (num_regions / regions.prod(dim=1, keepdim=True)) ** (1 / num_and_hashes) * regions

    regions = torch.round(regions * 3) / 3
    return rearrange(regions, "(h c) a -> c a h", h=num_heads)

def invert_permutation(perm: torch.Tensor) -> torch.Tensor:
    # More complicated than torch.argsort(perm, dim=-1) but has complexity O(n) vs O(n log n)
    arange = torch.arange(perm.shape[-1], device=perm.device).expand_as(perm)
    return torch.empty_like(perm).scatter_(-1, perm, arange)

def batched_index_select(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    last_dim = values.shape[-1]
    indices_expanded = rearrange(indices, "... -> ... 1").expand(*indices.shape, last_dim)
    return values.expand(*indices_expanded.shape[:-2], *values.shape[-2:]).gather(-2, indices_expanded)

@torch.no_grad()
def lsh_mapping(e2lsh, queries, keys):
    queries_hashed = e2lsh(queries)
    keys_hashed = e2lsh(keys)
    max_hash_shift = torch.max(queries_hashed.max(-1, keepdim=True).values, keys_hashed.max(-1, keepdim=True).values)
    min_hash_shift = torch.min(queries_hashed.min(-1, keepdim=True).values, keys_hashed.min(-1, keepdim=True).values)
    hash_shift = max_hash_shift - min_hash_shift
    return queries_hashed, keys_hashed, hash_shift

class E2LSH(nn.Module):
    def __init__(self, n_hashes, n_heads, dim, r=1):
        super(E2LSH, self).__init__()

        self.alpha = nn.Parameter(torch.normal(0, 1, (n_heads, dim, n_hashes)))
        self.alpha.requires_grad = False

    def forward(self, vecs):
        projection = torch.bmm(vecs, self.alpha)
        return projection.permute(2, 0, 1)