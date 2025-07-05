from datetime import datetime
import os
import math
import argparse
from tqdm import tqdm
from pathlib import Path
from itertools import combinations, product

import torch
from torch_geometric.transforms import BaseTransform
import numpy as np
import pandas as pd
from torch_geometric.data import InMemoryDataset
from torch_geometric.nn import radius_graph
from torch_geometric.utils import to_undirected, remove_self_loops
from joblib import Parallel, delayed
from utils import calculate_cluster_connectivity, save_cluster_connectivity

# Assumes phi --> cos/sin transform is used (default)
FEATURE_NAMES = ("sinphi","cosphi","eta","pt","r","x","y","z","dphi","dr_0","dx_0","dy_0","dz_0","dphi_0","dr_1","dx_1","dy_1","dz_1","dr_LS","dx_LS","dy_LS","dz_LS","layer_0","layer_1")
ROOT_PATH = Path(__file__).parent / "data"
CPU_CORES_FOR_MP = 8

class TrackingPileupTransform(BaseTransform):
    # placeholder transform function
    def __call__(self, data):
        return data

def get_new_idx_split(dataset):
    sorted_evtid = dataset.evtid.argsort()
    dataset_len = len(dataset)

    split = {"train": 0.8, "valid": 0.1, "test": 0.1}
    n_train = int(dataset_len * split["train"])
    n_valid = int(dataset_len * split["valid"])

    idx = sorted_evtid
    train_idx = idx[:n_train]
    valid_idx = idx[n_train : n_train + n_valid]
    test_idx = idx[n_train + n_valid :]
    return {"train": train_idx, "valid": valid_idx, "test": test_idx}


class TrackingPileup(InMemoryDataset):
    def __init__(self, task="tracking", truncate=None, num_events=None, graph_radius=1.0, graph_k_neighbors=None, phi_transform=True, LDA_path=None, use_LDA=True, mask=True, conn=False, pt_log=True, pileup_density="200", **kwargs):
        self.task = task
        self.pileup_density = str(pileup_density)
        self.root_path = ROOT_PATH
        self.feature_names = FEATURE_NAMES

        self.truncate = truncate
        self.num_events = num_events
        self.graph_radius = graph_radius
        self.graph_k_neighbors = graph_k_neighbors
        self.phi_transform = phi_transform
        self.use_lda = use_LDA
        self.mask = mask
        self.conn = conn
        self.pt_log = pt_log
        self.LDA_path = LDA_path if LDA_path is not None else Path(__file__).parent / f"lda/LDA_pu{self.pileup_density}.pt"

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
        if self.num_events is not None:
            fname = f"data-{self.num_events}.pt"
            processed_file_path = Path(self.processed_dir) / fname
        elif self.truncate is not None:
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

        # In case file data-n.pt for n = num_events is missing, use n to truncate and generate it
        if self.truncate is None and self.num_events is not None:
            self.truncate = self.num_events

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
        sel = self.truncate if self.truncate is not None else self.num_events
        if sel is not None:
            return [f"data-{sel}.pt"]
        size = len(os.listdir(self.raw_dir))
        return [f"data-{size}.pt"]

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
        
        if self.task == 'tracking' and self.mask:
            MASK_USES_ONLY_TRUE_TRACKS = True       # False will mix in some fake LS with the true, based on ratio (default=True, i.e. only use LS with sim-track label)
            MASKED_CLASS_BALANCE_RATIO = 2          # Class balance/mixing ratio if above is False (default=2 --> 2 fake LS for every true LS)
            
            if MASK_USES_ONLY_TRUE_TRACKS:
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
            if pid_item != -1:
                track_length_dict[pid_item] = count.item()
        tracklen = torch.zeros(data.particle_id.shape[0], dtype=torch.long)
        for i, pid in enumerate(data.particle_id):
            pid_item = pid.item()
            if pid_item == -1:
                tracklen[i] = 0         # Fake hits have track length 0
            else:
                tracklen[i] = track_length_dict.get(pid_item, 0)
        data.tracklen = tracklen

        if self.task == 'tracking':
            # radius-graph knn edge construction for infoNCE
            data.point_pairs_index = gen_point_pairs(data, radius=self.graph_radius, k_max_neighbors=self.graph_k_neighbors, conn=self.conn)
        return data

    def get_idx_split(self, dataset_len):
        self.split = {"train": 0.8, "valid": 0.1, "test": 0.1}
        n_train = int(dataset_len * self.split["train"])
        n_valid = int(dataset_len * self.split["valid"])

        idx = np.arange(dataset_len)
        train_idx = idx[:n_train]
        valid_idx = idx[n_train : n_train + n_valid]
        test_idx = idx[n_train + n_valid :]
        return {"train": train_idx, "valid": valid_idx, "test": test_idx}
    
def create_point_pairs_from_clusters(cluster_ids, event_id, nearby_point_pairs, compute_connectivity=False):
    unique_cluster_ids = torch.unique(cluster_ids)

    point_pairs = []
    connectivity_metrics = []
    
    for cluster_id in unique_cluster_ids:
        same_cluster_indices = (cluster_ids == cluster_id).nonzero().flatten()

        if cluster_id == -1 or same_cluster_indices.shape[0] <= 1:
            continue

        if compute_connectivity:
            connectivity = calculate_cluster_connectivity(same_cluster_indices, nearby_point_pairs)
            connectivity_metrics.append((cluster_id.item(), connectivity))

        cluster_nearby_points = nearby_point_pairs[1][torch.isin(nearby_point_pairs[0], same_cluster_indices)].unique()

        neg_pairs = torch.tensor(list(product(same_cluster_indices, cluster_nearby_points))).T
        point_pairs.append(neg_pairs)

        pos_pairs = torch.tensor(list(combinations(same_cluster_indices, 2))).T
        point_pairs.append(pos_pairs)

    point_pairs = torch.cat(point_pairs, dim=-1)
    
    if compute_connectivity:
        save_cluster_connectivity(connectivity_metrics, metrics_dir,event_id)
    
    return point_pairs

def gen_point_pairs(data, radius, k_max_neighbors=None, conn=False):
    if k_max_neighbors is None:     # default to 1/5*sqrt(N)
        k_max_neighbors = round(0.2*math.sqrt(data.particle_id[data.y==1].shape[0]))
    nearby_point_pairs = to_undirected(radius_graph(data.pos, r=radius, loop=False, max_num_neighbors=k_max_neighbors))
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

    parser.add_argument("-r", "--graph_radius", type=float, default=0.2, help="Radius hyperparameterto build radius_graphs in tracking task.")
    parser.add_argument("-k", "--graph_k_neighbors", type=int, default=None, help="Max number of neighbors for radius_graph in tracking task (if not specified, uses 1/5*sqrt(N)).")
    parser.add_argument("-c","--connectivity", action="store_true", help="Calculate and save connectivity metrics (--> data/cluster_analysis/)")

    parser.add_argument("-l", "--LDA_path", type=str, default=None, help="Ignore usually unless need to specify custom path to LDA data -- usually done automatically based on pu")
    parser.add_argument("-n", "--num_events", type=int, default=None, help="Ignore -- specified in cfg to indicate which data-n.pt file to use")
    parser.add_argument("-nolda", "--no_LDA", action="store_true", help="Use raw x features instead of LDA transformed features (default is to use LDA)")
    parser.add_argument("-nomask","--no_mask", action="store_true", help="Include fake line segments in edge construction for tracking (default is to use only true tracks)")
    parser.add_argument("-rawphi", "--phi_transform", action="store_false", help="Use raw phi and do NOT transform to sin and cos (default is phi --> sin/cos to fix wrap-around issue)")
    parser.add_argument("-rawpt","--pt_log", action="store_false", help="Use raw pt values instead of log pt transformation (default is log (pt - min(pt) + 1))")

    args = parser.parse_args()

    if args.truncate is not None:
        print(args.truncate)
        prefix_for_connectivity = args.truncate
    if args.num_events is not None:
        print(args.num_events)
        prefix_for_connectivity = args.num_events
    print(f"Using pileup {args.pileup_density}")
    assert int(args.pileup_density) in [50, 100, 200]
    if args.pt_log:
        print("Transforming pt to log(pt)")
    else:
        print("Using raw pt")
    if args.connectivity:
        try:
            prefix_for_connectivity
        except:
            prefix_for_connectivity = "all"
        timestamp = datetime.now().strftime("%H%M%S")
        metrics_dir = os.getcwd() + f"/data/cluster_analysis/evts={prefix_for_connectivity}_pu{args.pileup_density}_k={args.graph_k_neighbors if args.graph_k_neighbors else 'auto'}_r={args.graph_radius}_{timestamp}"
        metrics_dir = Path(metrics_dir)
        metrics_dir.mkdir(parents = True, exist_ok=True)
        print(f"Connectivity metrics will be saved to {metrics_dir}")

    dataset = TrackingPileup(task=args.task, truncate=args.truncate, num_events = args.num_events, pileup_density=args.pileup_density,
                        graph_radius=args.graph_radius, graph_k_neighbors=args.graph_k_neighbors, phi_transform=args.phi_transform,
                        LDA_path=args.LDA_path, use_LDA=not args.no_LDA, mask=not args.no_mask, conn=args.connectivity, pt_log=args.pt_log)