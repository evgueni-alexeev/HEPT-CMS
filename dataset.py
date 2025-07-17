"""
Main script to build datasets, from graph_*.pt event files saved in data/raw. To make those first, look at `data/process_ntuple.py`.
Run this through command line as below (default values are defined after module imports). Full list of arguments and their defaults is in *main* at the bottom.

Basic Usage:
1) Generate PILEUP dataset (first step, for real vs fake LS processing). By default will use ALL events in data/raw, set '-t <N>' to use the first N events:
    python dataset.py -d pileup -t <opt: use first N events> -pu <pileup_density = [50,100,200]>

2) (this step is done manually) Once pileup model finishes training to our liking, we need to save a copy to ~/eval/pileup/<checkpoint_folder>/best.ckpt.
To do this, copy logs/pileup/csv/version_XXX folder to that eval, change name to <checkpoint_folder>, something like 'pu200_0.409p' for e.g. 40.9% prec @ 99% recall,
then pick one of saved ckpt files in <checkpoint_folder>/checkpoints (usually the best val_fb or best val_loss), and copy/paste it as 'best.ckpt' into <checkpoint_folder>.

3) Filter with model, & generate TRACKING dataset (learns high-dim embedding space where LS from the same track are close together). Once there exists a trained model
 saved as ~/eval/pileup/<checkpoint_folder>/best.ckpt, run dataset.py again to make filtered tracking dataset as follows:
    python dataset.py -d tracking -t ... -pu ... -f -ckpt <checkpoint_folder>

"""

import warnings
# Python 3.9 superflous warning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="networkx backend defined more than once: nx-loopback")

from datetime import datetime
import os
import math
import argparse
from tqdm import tqdm
from pathlib import Path
from itertools import combinations, product
import sys
import yaml
from torch_geometric.data import Batch
from sklearn.metrics import precision_recall_curve

import torch
from torch_geometric.transforms import BaseTransform
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import radius_graph
from torch_geometric.utils import to_undirected, remove_self_loops
from joblib import Parallel, delayed
from eval.tracking.cluster_analysis import (calculate_cluster_connectivity, save_cluster_connectivity, calculate_cluster_noise_contamination, save_cluster_noise_contamination)

# Assumes phi --> cos/sin transform is used (default)
FEATURE_NAMES = ("sinphi","cosphi","eta","pt","r","x","y","z","dphi","dr_0","dx_0","dy_0","dz_0","dphi_0","dr_1","dx_1","dy_1","dz_1","dr_LS","dx_LS","dy_LS","dz_LS","layer_0","layer_1")
ROOT_PATH = Path(__file__).parent / "data"
PILEUP_MODEL = "pu200_0.409p"
TRAIN_VAL_TEST_SPLIT = {"train": 0.8, "valid": 0.1, "test": 0.1}

DEFAULT_RADIUS = 0.15
DEFAULT_K_MAX_NEIGHBORS = 1.0
DEFAULT_RECALL = 0.99

CPU_CORES_FOR_MP = 8

class TrackingPileupTransform(BaseTransform):
    # Placeholder transform function that is called during fit/test
    def __call__(self, data):
        return data

class BuildInfoNCEEdges(BaseTransform):
    # Add edges to a dataset post-hoc (i.e. after filtering with a pileup model)
    def __init__(self, radius = DEFAULT_RADIUS, k_max_neighbors = DEFAULT_K_MAX_NEIGHBORS, conn = False, metrics_dir= None):
        self.radius = radius
        self.k_max_neighbors = k_max_neighbors
        self.conn = conn
        self.metrics_dir = metrics_dir

    def __call__(self, data):
        # Skip if edges already present
        if self.conn and self.metrics_dir is not None:
            global metrics_dir
            metrics_dir = self.metrics_dir

        if getattr(data, "point_pairs_index", None) is None or data.point_pairs_index.numel() == 0:
            data.point_pairs_index = gen_point_pairs(
                data,
                radius=self.radius,
                k_max_neighbors=self.k_max_neighbors,
                conn=self.conn,
            )
        return data

def add_edges_to_dataset(input_pt, output_pt, radius = DEFAULT_RADIUS, k_max_neighbors = DEFAULT_K_MAX_NEIGHBORS, conn = False, n_jobs = 1):
    # Utility function to add edges to dataset, using above transform
    # Works with any *Data* object with the `pos`, `particle_id`, `y`,... attributes expected by `gen_point_pairs`

    input_pt = Path(input_pt)
    output_pt = Path(output_pt)

    print(f"Loading dataset from {input_pt}")
    data, slices, idx_split = torch.load(input_pt, weights_only=False)

    metrics_dir_local = None
    if conn:
        metrics_dir_local = output_pt.parent / "cluster_analysis" / output_pt.stem
        metrics_dir_local.mkdir(parents=True, exist_ok=True)
        print(f"Connectivity metrics will be written to {metrics_dir_local}")

    # Simple wrapper
    class _Dataset(InMemoryDataset):
        def __init__(self, _data, _slices, transform=None):
            super().__init__(root=None, transform=transform)
            self.data, self.slices = _data, _slices

    transform = BuildInfoNCEEdges(radius=radius, k_max_neighbors=k_max_neighbors, conn=conn, metrics_dir=metrics_dir_local)
    ds = _Dataset(data, slices, transform=transform)
    processed = [ds[i] for i in tqdm(range(len(ds)), desc="Building edges")]
    # processed = Parallel(n_jobs=n_jobs)(
    #     delayed(ds.__getitem__)(i) for i in tqdm(range(len(ds)), desc="Building edges")
    # )

    new_data, new_slices = ds.collate(processed)
    torch.save((new_data, new_slices, idx_split), output_pt)
    print(f"Saved augmented dataset to {output_pt}")

def get_new_idx_split(dataset):
    sorted_evtid = dataset.evtid.argsort()
    dataset_len = len(dataset)

    # split = {"train": 0.8, "valid": 0.1, "test": 0.1}
    split = TRAIN_VAL_TEST_SPLIT
    n_train = int(dataset_len * split["train"])
    n_valid = int(dataset_len * split["valid"])

    idx = sorted_evtid
    train_idx = idx[:n_train]
    valid_idx = idx[n_train : n_train + n_valid]
    test_idx = idx[n_train + n_valid :]
    return {"train": train_idx, "valid": valid_idx, "test": test_idx}

class TrackingPileup(InMemoryDataset):
    def __init__(self, task="tracking", truncate=None, num_events=None, pileup_density = "200", use_LDA=True, LDA_path=None, phi_transform=True,  pt_log = True,
                graph_radius=DEFAULT_RADIUS, graph_k_neighbors=DEFAULT_K_MAX_NEIGHBORS, mask=True, conn=False, filtered = True, recall = DEFAULT_RECALL, pileup_model = PILEUP_MODEL, **kwargs):
                
        self.task = task
        self.pileup_density = str(pileup_density)
        self.root_path = ROOT_PATH
        self.feature_names = FEATURE_NAMES

        self.truncate = truncate
        self.filtered = filtered and (task == "tracking")
        self.recall = recall
        self.pileup_model_path = Path(__file__).parent / f"eval/pileup/{pileup_model}"
        self.num_events = num_events
        self.graph_radius = graph_radius
        self.graph_k_neighbors = graph_k_neighbors
        self.phi_transform = phi_transform
        self.use_lda = use_LDA
        self.mask = mask
        self.conn = conn
        self.pt_log = pt_log

        self.filter_masks_path = None
        if self.filtered:
            suffix = int(self.recall * 1000)
            mask_dir = self.pileup_model_path / "event_masks"  # stored masks for a given recall
            mask_dir.mkdir(parents=True, exist_ok=True)
            filter_masks_path = mask_dir / f"event_masks_r{suffix}.pt"

            # Determine the set of event IDs required for this run (based on raw files)
            required_evtids = []
            try:
                all_point_clouds = os.listdir(self.raw_dir)
                all_point_clouds = [f for f in all_point_clouds if f.endswith('.pt') and os.path.isfile(os.path.join(self.raw_dir, f))]
                all_point_clouds.sort(key=lambda s: int(s.split('_')[1].split('.')[0]))
                sel = self.truncate if self.truncate is not None else self.num_events
                if sel is not None and sel <= len(all_point_clouds):
                    all_point_clouds = all_point_clouds[:sel]
                required_evtids = {get_event_id_sector_from_str(f)[0] for f in all_point_clouds}
            except FileNotFoundError:
                # raw_dir may not exist yet if we're only loading an existing processed file
                required_evtids = set()

            # If mask file doesn't exist or is missing some required events, (re)generate masks
            regenerate_masks = True
            missing_evtids = required_evtids
            if filter_masks_path.exists():
                try:
                    existing_masks = torch.load(filter_masks_path, weights_only=False)
                    existing_evtids = {eid for eid, _ in existing_masks}
                    missing_evtids = required_evtids - existing_evtids
                    regenerate_masks = len(missing_evtids) > 0
                except Exception:
                    regenerate_masks = True
            if regenerate_masks:
                print(f"Generating masks for event IDs: {sorted(missing_evtids)}")
                # Build/extend mask file covering at least the events needed for this run
                self.generate_filter_masks(event_ids=list(missing_evtids), checkpoint_dir=self.pileup_model_path)
            else:
                print("All required event masks already exist - skipping mask generation.")

            self.filter_masks_path = filter_masks_path

        self.LDA_path = LDA_path if LDA_path is not None else Path(__file__).parent / f"lda/LDA_pu{self.pileup_density}.pt"

        self.filter_masks = None
        if self.filter_masks_path is not None:
            self.filter_masks = torch.load(self.filter_masks_path, weights_only=False)
            self.filter_masks_dict = {event_id: mask for event_id, mask in self.filter_masks}

        if self.use_lda:
            if not self.num_events: print(f"Using LDA transformed features from {self.LDA_path}")
            self.LDA_data = torch.load(self.LDA_path, weights_only=False)
            self.LDA_features = self.LDA_data['feature_names']
            self.LDA_scalings = self.LDA_data['scalings_matrix']
            assert self.LDA_scalings.shape[0] == len(self.feature_names), f"LDA features and dataset features are different"
            assert self.LDA_scalings.shape[1] >=3, f"LDA scalings have less than 3 components"
            if self.LDA_scalings.shape[1] > 7:
                self.LDA_scalings = self.LDA_scalings[:,:7]
            
        super(TrackingPileup, self).__init__(str(self.root_path), transform=kwargs.get("transform", None), pre_transform=None)
        
        processed_file_path = None
        if self.filtered:
            processed_file_path = self.processed_paths[0]
        elif self.num_events:
            fname = f"data-{self.num_events}.pt"
            processed_file_path = Path(self.processed_dir) / fname
        elif self.truncate:
            fname = f"data-{self.truncate}.pt"
            processed_file_path = Path(self.processed_dir) / fname
        else:
            processed_file_path = self.processed_paths[0]
        
        self.data, self.slices, self.idx_split = torch.load(processed_file_path,weights_only=False)

        self.idx_split = get_new_idx_split(self)
        self.x_dim = self._data.x.shape[1]
        
        if hasattr(self._data, 'coords') and self._data.coords is not None:
            self.coords_dim = self._data.coords.shape[1]
        else:
            self.coords_dim = 7

        # Sync self.truncate to actual dataset size when it was not explicitly provided
        if self.truncate is None:
            if self.num_events is not None:
                self.truncate = self.num_events
            else:
                # Infer from loaded data length (all raw files were processed)
                self.truncate = len(self)

    @property
    def raw_dir(self):
        return self.root_path / "raw" / f"pu{self.pileup_density}"

    @property
    def processed_dir(self):
        return self.root_path / "processed" / self.task / f"pu{self.pileup_density}"

    @property
    def raw_file_names(self):
        return ["graph_12.pt"]

    @property
    def processed_file_names(self):
        total_raw = len([f for f in os.listdir(self.raw_dir) if f.endswith('.pt')])
        requested = None
        if self.truncate is not None:
            requested = self.truncate
        elif self.num_events is not None:
            requested = self.num_events

        n_events = requested if requested is not None and requested <= total_raw else total_raw
        return [f"data-{n_events}.pt"]

    def process(self):
        # List all .pt files in the specified directory (not including subdirs)
        all_point_clouds = os.listdir(self.raw_dir)
        all_point_clouds = [f for f in all_point_clouds if f.endswith('.pt') and os.path.isfile(os.path.join(self.raw_dir, f))]
        all_point_clouds.sort(key=lambda s: int(s.split('_')[1].split('.')[0]))

        sel = self.truncate if self.truncate is not None else self.num_events
        if sel is not None and sel <= len(all_point_clouds):
            all_point_clouds = all_point_clouds[:sel]

        data_list = Parallel(n_jobs=CPU_CORES_FOR_MP)(
            delayed(self.process_point_cloud)(point_cloud) for point_cloud in tqdm(all_point_clouds)
        )

        data, slices = self.collate(data_list)

        idx_split = self.get_idx_split(len(data_list))
        torch.save((data, slices, idx_split), self.processed_paths[0])
    
    def transform_data(self, data):
        phis = data.x[:,0]
        others = data.x[:,1:]
        sin_phi = torch.sin(phis)
        cos_phi = torch.cos(phis)
        data.x = torch.cat([sin_phi.unsqueeze(1), cos_phi.unsqueeze(1), others], dim=1)
        return data

    # Main loop for processing a single event/point cloud
    def process_point_cloud(self, point_cloud):
        evtid, sector = get_event_id_sector_from_str(point_cloud)
        data = torch.load(Path(self.raw_dir) / point_cloud,weights_only=False)
        data = preprocess_data(data, evtid, phi_transform=self.phi_transform, topk_pt=0, pt_log=self.pt_log)
        data.particle_id = data.particle_id.long()
        
        if self.task == 'tracking' and self.mask:
            MASK_USES_ONLY_TRUE_TRACKS = True       # False will mix in some fake LS with the true, based on ratio (default=True, i.e. only use LS with sim-track label)
            MASKED_CLASS_BALANCE_RATIO = 2          # Class balance/mixing ratio if above is False (default=2 --> 2 fake LS for every true LS)
            if self.filtered and evtid in self.filter_masks_dict:            # If using masks after filtering with pileup model
                filter_mask = self.filter_masks_dict[evtid]
                data.x = data.x[filter_mask]
                data.particle_id = data.particle_id[filter_mask]
                data.pt = data.pt[filter_mask]
                data.y = data.y[filter_mask]
            elif MASK_USES_ONLY_TRUE_TRACKS:
                print(f"USING ONLY TRUE TRACKS {evtid}")
                data.x = data.x[data.y==1]
                data.particle_id = data.particle_id[data.y==1]
                data.pt = data.pt[data.y==1]
                data.y = data.y[data.y==1]
            else:
                pos_mask = (data.y == 1)
                neg_mask = ~pos_mask
                pos_idx = pos_mask.nonzero(as_tuple = False).view(-1)
                neg_idx = neg_mask.nonzero(as_tuple = False).view(-1)
                n_pos = pos_idx.numel()
                perm = torch.randperm(neg_idx.numel())[:int(MASKED_CLASS_BALANCE_RATIO*n_pos)]
                neg_sample = neg_idx[perm]
                keep = torch.cat([pos_idx, neg_sample], dim=0)
                keep = keep[torch.randperm(keep.numel())]
                data.x = data.x[keep]
                data.particle_id = data.particle_id[keep]
                data.y = data.y[keep]
                data.pt = data.pt[keep]

        df = get_dataframe(data, evtid, self.feature_names)
        
        if self.use_lda:
            coord_transform = torch.matmul(data.x, self.LDA_scalings)
            stdev, means = torch.std_mean(coord_transform, dim=0)
            coord_transform = (coord_transform - means)/stdev            
            data.pos = coord_transform[:,:3]
            data.coords = coord_transform      
        else:
            sinphi = df.sinphi
            cosphi = df.cosphi
            eta = df.eta
            data.pos = torch.tensor([eta, sinphi, cosphi]).T
            data.coords = torch.cat([data.pos, data.x[:, 4:8]], dim=-1)

        data.evtid = torch.tensor([evtid]).long()
        data.layer = torch.tensor(df.layer_0.astype(int))
        data.reconstructable = torch.ones(data.x.size(0))       # placeholder for now (all tracks reconstructable)
        data.s = torch.tensor([sector]).long()                  # placeholder for sector information

        # Calculate imbalance ratio for this event (data.imb)
        num_positive = (data.y == 1).sum().item()
        num_negative = (data.y == 0).sum().item()
        if num_positive > 0:
            imbalance_ratio = num_negative / num_positive
        else:
            imbalance_ratio = float('inf')
        data.imb = torch.tensor([imbalance_ratio]).float()

        # Calculate track lengths for each data point (data.tracklen)
        unique_particle_ids, counts = torch.unique(data.particle_id, return_counts=True)
        track_length_dict = {}
        for pid, count in zip(unique_particle_ids, counts):
            pid_item = pid.item()
            if pid_item >= 0:
                track_length_dict[pid_item] = count.item()
        tracklen = torch.zeros(data.particle_id.shape[0], dtype=torch.long)
        for i, pid in enumerate(data.particle_id):
            pid_item = pid.item()
            if pid_item <= -1:
                tracklen[i] = 0         # Fake hits have track length 0
            else:
                tracklen[i] = track_length_dict.get(pid_item, 0)
        data.tracklen = tracklen

        if self.task == 'tracking':
            # radius-graph knn edge construction for infoNCE
            data.point_pairs_index = gen_point_pairs(data, radius=self.graph_radius, k_max_neighbors=self.graph_k_neighbors, conn=self.conn)
        return data

    def get_idx_split(self, dataset_len):
        # self.split = {"train": 0.8, "valid": 0.1, "test": 0.1}
        self.split = TRAIN_VAL_TEST_SPLIT
        n_train = int(dataset_len * self.split["train"])
        n_valid = int(dataset_len * self.split["valid"])

        idx = np.arange(dataset_len)
        train_idx = idx[:n_train]
        valid_idx = idx[n_train : n_train + n_valid]
        test_idx = idx[n_train + n_valid :]
        return {"train": train_idx, "valid": valid_idx, "test": test_idx}

    def generate_filter_masks(self, event_ids, checkpoint_dir):
        """Run the pile-up model to create boolean masks for the specified event IDs only."""
        from importlib import import_module
        fm = import_module("eval.filter_model")
        event_ids = sorted(set(int(e) for e in event_ids))
        if len(event_ids) == 0:
            print("No event IDs provided for mask generation - nothing to do.")
            return

        temp_name = f"data_temp_{len(event_ids)}.pt"
        pileup_pt = self.root_path / f"processed/pileup/pu{self.pileup_density}/{temp_name}"

        # Build dataset for the specified events only
        raw_dir = self.root_path / "raw" / f"pu{self.pileup_density}"
        data_list = []
        if self.use_lda and not hasattr(self, "LDA_scalings"):
            self.LDA_path = getattr(self, "LDA_path", Path(__file__).parent / f"lda/LDA_pu{self.pileup_density}.pt")
            print(f"Loading LDA scalings from {self.LDA_path} for temporary dataset")
            lda_data_tmp = torch.load(self.LDA_path, weights_only=False)
            lda_scalings_tmp = lda_data_tmp["scalings_matrix"]
            if lda_scalings_tmp.shape[1] > 7:
                lda_scalings_tmp = lda_scalings_tmp[:, :7]
            self.LDA_scalings = lda_scalings_tmp

        for evtid in event_ids:
            raw_file = raw_dir / f"graph_{evtid}.pt"
            if not raw_file.exists():
                print(f"Warning: raw file {raw_file} not found - skipping")
                continue
            data = torch.load(raw_file, weights_only=False)
            data = preprocess_data(data, evtid, phi_transform=self.phi_transform, topk_pt=0, pt_log=self.pt_log)

            # Build coords/pos similar to pileup processing
            df = get_dataframe(data, evtid, self.feature_names)
            if self.use_lda:
                coord_transform = torch.matmul(data.x, self.LDA_scalings)
                stdev, means = torch.std_mean(coord_transform, dim=0)
                coord_transform = (coord_transform - means) / stdev
                data.pos = coord_transform[:, :3]
                data.coords = coord_transform
            else:
                sinphi = df.sinphi
                cosphi = df.cosphi
                eta = df.eta
                data.pos = torch.tensor([eta, sinphi, cosphi]).T
                data.coords = torch.cat([data.pos, data.x[:, 4:8]], dim=-1)

            data.evtid = torch.tensor([evtid]).long()
            data.layer = torch.tensor(df.layer_0.astype(int))
            data.reconstructable = torch.ones(data.x.size(0))
            data.s = torch.tensor([0]).long()

            # imbalance ratio (not strictly needed but keeps structure consistent)
            num_positive = (data.y == 1).sum().item()
            num_negative = (data.y == 0).sum().item()
            imbalance_ratio = num_negative / num_positive if num_positive > 0 else float('inf')
            data.imb = torch.tensor([imbalance_ratio]).float()

            # track length info (optional)
            unique_particle_ids, counts = torch.unique(data.particle_id, return_counts=True)
            track_length_dict = {pid.item(): cnt.item() for pid, cnt in zip(unique_particle_ids, counts) if pid.item() >= 0}
            tracklen = torch.zeros(data.particle_id.shape[0], dtype=torch.long)
            for i, pid in enumerate(data.particle_id):
                pid_item = pid.item()
                tracklen[i] = 0 if pid_item <= -1 else track_length_dict.get(pid_item, 0)
            data.tracklen = tracklen

            data_list.append(data)

        # Collate and save temp dataset
        if len(data_list) == 0:
            print("No valid events found - aborting mask generation.")
            return

        data_combined, slices = self.collate(data_list)
        pileup_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save((data_combined, slices, {}), pileup_pt)

        data, slices, _ = torch.load(pileup_pt, weights_only=False)
        pu_dataset = fm.EventDataset(data, slices)

        sample = pu_dataset[0]
        in_dim, coords_dim = sample.x.shape[1], sample.coords.shape[1]

        ckpt_path = checkpoint_dir / "best.ckpt"
        hparams_path = checkpoint_dir / "hparams.yaml"

        model = fm._load_model(ckpt_path, hparams_path, in_dim, coords_dim, task="pileup")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        y_true_all, y_prob_all, preds_per_event = [], [], []
        print(f"Filtering {len(event_ids)} events @ {self.recall} recall")
        for evt in tqdm(pu_dataset):
            batch_evt = Batch.from_data_list([evt]).to(device)
            with torch.no_grad():
                logits = model(batch_evt).squeeze()
                probs = torch.sigmoid(logits).cpu()
            preds_per_event.append(probs)
            y_prob_all.append(probs)
            y_true_all.append(evt.y.cpu())

        y_true_cat = torch.cat(y_true_all)
        y_prob_cat = torch.cat(y_prob_all)

        mask_dir = checkpoint_dir / "event_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        fm.export_masks(pu_dataset, preds_per_event, self.recall, mask_dir, len(event_ids), y_true_cat=y_true_cat, y_prob_cat=y_prob_cat)
        
        if pileup_pt.exists():
            try:
                os.remove(pileup_pt)
            except OSError as e:
                print(f"Warning: could not delete temporary pileup dataset {pileup_pt}: {e}")

        # Clean up GPU memory and model from RAM
        del model, pu_dataset, y_true_all, y_prob_all, preds_per_event, y_true_cat, y_prob_cat
        if torch.cuda.is_available(): torch.cuda.empty_cache()

def create_point_pairs_from_clusters(cluster_ids, event_id, nearby_point_pairs, compute_connectivity=False):
    fake_counter = 2
    neg_label_shift = 1_000_000

    if isinstance(event_id, torch.Tensor):
        event_id_val = int(event_id.item())
    else:
        event_id_val = int(event_id)
    
    unique_cluster_ids = torch.unique(cluster_ids)

    point_pairs = []
    connectivity_metrics = []
    noise_metrics = []
    
    for cluster_id in unique_cluster_ids:
        same_cluster_indices = (cluster_ids == cluster_id).nonzero().flatten()

        if cluster_id == -1 or same_cluster_indices.shape[0] <= 1:
            continue

        cluster_nearby_points = nearby_point_pairs[1][torch.isin(nearby_point_pairs[0], same_cluster_indices)].unique()

        # Re-label unassigned fake hits
        fake_mask = cluster_ids[cluster_nearby_points] == -1
        fake_indices = cluster_nearby_points[fake_mask]

        if fake_indices.numel() > 0:
            # All fake hits in this cluster get the same negative track ID
            common_fake_id = -(event_id_val * neg_label_shift + fake_counter)
            cluster_ids[fake_indices] = common_fake_id
            fake_counter += 1
        
        if compute_connectivity:
            connectivity = calculate_cluster_connectivity(same_cluster_indices, nearby_point_pairs)
            track_length = len(same_cluster_indices)
            connectivity_metrics.append((cluster_id.item(), connectivity, track_length))
            noise_count, total_neighbors, signal_points = calculate_cluster_noise_contamination(same_cluster_indices, nearby_point_pairs, cluster_ids)
            noise_metrics.append((cluster_id.item(), noise_count, total_neighbors, signal_points))

        neg_pairs = torch.tensor(list(product(same_cluster_indices, cluster_nearby_points))).t()
        point_pairs.append(neg_pairs)

        pos_pairs = torch.tensor(list(combinations(same_cluster_indices, 2))).t()
        point_pairs.append(pos_pairs)

    point_pairs = torch.cat(point_pairs, dim=-1)
    
    if compute_connectivity:
        save_cluster_noise_contamination(noise_metrics, metrics_dir, event_id)
        save_cluster_connectivity(connectivity_metrics, metrics_dir, event_id)
    
    return point_pairs

def gen_point_pairs(data, radius, k_max_neighbors=1.0, conn=False):
    if k_max_neighbors is None:
        k_max_neighbors = 1.0       # k ~ 30 for base case (N ~ 10k true + 10-15k fake LS at 0.99 recall)
    
    base_factor = 0.0015
    k_max = max(1, round(base_factor * k_max_neighbors * data.particle_id.shape[0]))    # cluster connectivity stays roughly constant if we scale k_max by N)
    
    nearby_point_pairs = to_undirected(radius_graph(data.pos, r=radius, loop=False, max_num_neighbors=k_max))
    point_pairs = create_point_pairs_from_clusters(data.particle_id, data.evtid, nearby_point_pairs, compute_connectivity=conn)
    point_pairs = point_pairs.long()
    point_pairs = remove_self_loops(to_undirected(point_pairs))[0]
    return point_pairs

def get_dataframe(evt, evtid, feature_names):
    to_df = {"evtid": evtid}
    for i, n in enumerate(feature_names):
        to_df[n] = evt.x[:, i]
    return pd.DataFrame(to_df)

def get_event_id_sector_from_str(name: str) -> tuple[int, int]:
    evtid = int(name.split("_")[1][:-3])
    sectorid = int(0)
    return evtid, sectorid

def preprocess_data(data, evtid, phi_transform=True, topk_pt=0, pt_log = False):
    n_pts, n_feats = data.x.size()

    # phi --> sin/cos phi
    if n_feats == 23 and phi_transform:
        phis = data.x[:,0]
        others = data.x[:,1:]
        sin_phi = torch.sin(phis)
        cos_phi = torch.cos(phis)
        data.x = torch.cat([sin_phi.unsqueeze(1), cos_phi.unsqueeze(1), others], dim=1)
        n_pts, n_feats = data.x.size()

    # pt --> log(pt - min(pt) + 1), exclude the largest k outliers from std/mean calc
    data.pt = data.x[:,3].detach().clone()
    if topk_pt != 0:
        k = [0]*n_feats
        k[3] = topk_pt
        assert len(k)==n_feats and all(isinstance(i,int) for i in k)
        topk_vals = torch.tensor([torch.std_mean(torch.topk(data.x[:, i], n_pts - k[i], largest=False).values) for i in range(n_feats)])
        stdevs,means = topk_vals[:,0], topk_vals[:,1]
    else:
        if pt_log:
            logpt = torch.log(data.x[:, 3] - torch.min(data.x[:, 3]) + 1)
            data.x[:, 3] = logpt
        stdevs,means = torch.std_mean(data.x, dim=0)

    # Standardize all but a few columns
    no_norm_list = [-1,-2, 0, 1, 2] if (topk_pt != 0) else [-1,-2,0,1,2,3]  #layer1, layer0, sinphi, cosphi, eta, pt
    for i in no_norm_list:
        means[i] = 0.0
        stdevs[i] = 1.0

    data.x = (data.x - means) / stdevs
    data.y = (data.particle_id != -1).long()

    # Prevent label collision between events
    label_shift = 1000000
    data.particle_id[data.y==1] += label_shift*evtid

    return data

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Build point clouds from raw data.")
    parser.add_argument("-d", "--task", type=str, default="pileup", choices=["tracking", "pileup"], help="Specify the dataset task: tracking or pileup")
    parser.add_argument("-pu", "--pileup_density", type=str, default="200", help="Pileup density (50, 100 or 200)")
    parser.add_argument("-t", "--truncate", type=int, default=None, help="For generating -- process first t graph_i.pt files in raw dir to make data-t.pt file. By default will process all files in raw.")

    parser.add_argument("-rad", "--graph_radius", type=float, default=DEFAULT_RADIUS, help="Radius hyperparameterto build radius_graphs in tracking task.")
    parser.add_argument("-k", "--graph_k_neighbors", type=float, default=DEFAULT_K_MAX_NEIGHBORS, help="Multiplier for base neighbor count (base 0.0015*N). Use 1 for default (k~30 after filtering).")
    parser.add_argument("-c","--connectivity", action="store_true", help="Calculate and save connectivity metrics (--> data/cluster_analysis/)")

    parser.add_argument("-l", "--LDA_path", type=str, default=None, help="Ignore usually unless need to specify custom path to LDA data -- usually done automatically based on pu")
    parser.add_argument("-n", "--num_events", type=int, default=None, help="Ignore -- specified in cfg to indicate which data-n.pt file to use")
    parser.add_argument("-nolda", "--no_LDA", action="store_true", help="Use raw x features instead of LDA transformed features (default is to use LDA)")
    parser.add_argument("-nomask","--no_mask", action="store_true", help="Include fake line segments in edge construction for tracking (default is to use only true tracks)")
    parser.add_argument("-rawphi", "--phi_transform", action="store_false", help="Use raw phi and do NOT transform to sin and cos (default is phi --> sin/cos to fix wrap-around issue)")
    parser.add_argument("-rawpt","--pt_log", action="store_false", help="Use raw pt values instead of log pt transformation (default is log (pt - min(pt) + 1))")

    parser.add_argument("-f", "--filtered", action="store_true", help="Run pile-up filtering before building tracking dataset")
    parser.add_argument("-r", "--recall", type=float, default=DEFAULT_RECALL, help="Recall target when filtering (e.g. 0.99)")
    parser.add_argument("-ckpt", "--pileup_model", type=str, default=PILEUP_MODEL, help="Checkpoint folder inside eval/pileup (defaults to pu<density>_0.409p)")

    args = parser.parse_args()

    print(f"Using pileup {args.pileup_density}")
    assert int(args.pileup_density) in [50, 100, 200]
    if args.pt_log:
        print("Transforming pt to log(pt - min(pt) + 1)")
    else:
        print("Using raw pt")

    sel_events = None
    if args.truncate is not None:
        sel_events = args.truncate
    if args.num_events is not None:
        sel_events = args.num_events

    # If neither truncate nor num_events provided, infer total event count for messaging and overwrite checks
    processed_base = ROOT_PATH / "processed" / args.task / f"pu{args.pileup_density}"
    target_file_path = None
    if sel_events is None:
        raw_dir_default = ROOT_PATH / "raw" / f"pu{args.pileup_density}"
        if raw_dir_default.exists():
            sel_events = len([f for f in os.listdir(raw_dir_default) if f.endswith('.pt')])
            target_file_path = processed_base / f"data-{sel_events}.pt"

    # Prompt before overwriting existing dataset file (only when script is executed directly)
    if target_file_path is not None and target_file_path.exists():
        try:
            if sys.stdin.isatty():
                resp = input(f"Dataset file '{target_file_path}' already exists. Overwrite? [y/n]: ").strip().lower()
                if resp != "y":
                    print("Exiting without overwriting the existing dataset file.")
                    sys.exit(0)
                else:
                    print("Overwriting existing dataset file...")
                    target_file_path.unlink()
            else:
                print(f"Dataset file '{target_file_path}' already exists and no TTY available to prompt. Exiting.")
                sys.exit(1)
        except KeyboardInterrupt:
            print("Operation cancelled by user. Exiting.")
            sys.exit(0)

    if args.connectivity:
        prefix_for_connectivity = sel_events if sel_events is not None else "all"
        timestamp = datetime.now().strftime("%H%M%S")
        metrics_dir = os.getcwd() + f"/data/cluster_analysis/evts={prefix_for_connectivity}_pu{args.pileup_density}_k={args.graph_k_neighbors if args.graph_k_neighbors else 'auto'}_r={args.graph_radius}_{timestamp}"
        metrics_dir = Path(metrics_dir)
        metrics_dir.mkdir(parents = True, exist_ok=True)
        print(f"Connectivity metrics will be saved to {metrics_dir}")

    dataset = TrackingPileup(task=args.task, truncate=args.truncate, num_events = args.num_events, pileup_density=args.pileup_density,
                        graph_radius=args.graph_radius, graph_k_neighbors=args.graph_k_neighbors, phi_transform=args.phi_transform,
                        LDA_path=args.LDA_path, use_LDA=not args.no_LDA, mask=not args.no_mask, conn=args.connectivity, pt_log=args.pt_log,
                        filtered=args.filtered, recall=args.recall, pileup_model=args.pileup_model)

    print(f"Finished building {args.task} dataset data-{len(dataset)}.pt")