import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[1]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from typing import List
import numpy as np
import argparse
import torch
import yaml
from torch_geometric.data import Batch, InMemoryDataset
from sklearn.metrics import precision_recall_curve

from dataset import add_edges_to_dataset
from model import Transformer

CHECKPOINT_FOLDER = "pu200_0.409p"
PILEUP_DENSITY = 200
PILEUP_NUM_EVENTS = 10
RECALL_TARGETS = [0.99]          #[0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]
SKIP_TRACKING = True       # override to skip construction of filtered tracking datasets
EXPORT_MASKS_ONLY = False        # export mask for recall target and exit

# for edge construction, if building tracking datasets
CONNECTIVITY_METRICS = False     # run and save connectivity metrics
KNN_MAX_NEIGHBORS = 1.0         # multiplier for base neighbor count (base 0.0015*N). Use 1 for default (k≈30 after filtering)
KNN_RADIUS = [0.15]             # specify 3D radius for euclidean knn radius graph (can be multiple)

CPU_CORES_FOR_MP = 8

class EventDataset(InMemoryDataset):
    def __init__(self, data, slices):
        super().__init__(root=None)
        self.data, self.slices = data, slices

def _load_model(ckpt_path: Path, hparams_path: Path, in_dim: int, coords_dim: int, task: str) -> torch.nn.Module:
    config = yaml.safe_load(hparams_path.open("r").read())
    model = Transformer(in_dim=in_dim, coords_dim=coords_dim, task=task, **config["model_kwargs"])

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state_dict, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #torch.device("cpu")
    return model.to(device).eval()

def _threshold_at_recall(y_true: torch.Tensor, y_scores: torch.Tensor, recall_target: float) -> float:
    precision, recall, thresh = precision_recall_curve(y_true.cpu().numpy(), y_scores.cpu().numpy())
    eligible = np.where(recall[1:] >= recall_target)[0]
    if len(eligible) == 0:
        print(f"Warning: recall never reaches {recall_target:.3f} - using threshold 0.0")
        return 0.0
    best_idx = eligible[-1]
    return float(thresh[best_idx])

def _filter_event(data_evt, probs_evt: torch.Tensor, thr: float):
    mask = probs_evt >= thr
    if mask.sum() == 0:
        mask[torch.argmax(probs_evt)] = True

    new_evt = data_evt.clone()
    num_nodes = int(mask.sum())

    for key in data_evt.keys():
        val = getattr(data_evt, key)
        if torch.is_tensor(val) and val.dim() >= 1 and val.size(0) == mask.size(0):
            setattr(new_evt, key, val[mask])
    new_evt.num_nodes = num_nodes
    return new_evt

def export_masks(dataset, preds_per_event, recall_target, output_dir, num_events, threshold = None, y_true_cat = None, y_prob_cat = None):
    """Export boolean masks per event for a given recall target."""
    # Compute threshold if not provided
    if threshold is None:
        if y_true_cat is None or y_prob_cat is None:
            raise ValueError("y_true_cat and y_prob_cat must be provided when threshold is None")
        threshold = _threshold_at_recall(y_true_cat, y_prob_cat, recall_target)
    event_masks = []
    for evt, p_evt in zip(dataset, preds_per_event):
        mask = p_evt >= threshold
        if mask.sum() == 0:
            mask[torch.argmax(p_evt)] = True
        event_id = evt.evtid.item() if hasattr(evt, "evtid") else 0
        event_masks.append((event_id, mask))

    suffix = int(recall_target * 1000)
    mask_file = output_dir / f"event_masks_r{suffix}.pt"

    # If aggregated file exists, merge – new masks overwrite existing entries for same event_id
    if mask_file.exists():
        try:
            existing = torch.load(mask_file, weights_only=False)
            mask_dict = {eid: m for eid, m in existing}
        except Exception:
            mask_dict = {}
    else:
        mask_dict = {}

    for eid, m in event_masks:
        mask_dict[eid] = m

    # Convert back to list sorted by evtid for consistency
    merged_masks = [(eid, mask_dict[eid]) for eid in sorted(mask_dict.keys())]
    torch.save(merged_masks, mask_file)
    return mask_file

def load_filtered_dataset(recall: float, num_events: int, ckpt_folder: str = CHECKPOINT_FOLDER, root: Path | None = None):
    if root is None:
        root = Path(__file__).resolve().parents[1]

    filtered_dir = root / "eval" / "pileup" / ckpt_folder / "filtered_data"
    fpath = filtered_dir / f"filtered_r{int(recall*1000)}_data-{num_events}.pt"
    if not fpath.exists():
        raise FileNotFoundError(f"Filtered dataset not found: {fpath}")

    data, slices, idx_split = torch.load(fpath, weights_only=False)

    class _EventDataset(InMemoryDataset):
        def __init__(self, _data, _slices):
            super().__init__(root=None)
            self.data, self.slices = _data, _slices

    return _EventDataset(data, slices), idx_split

def build_tracking_dataset(recall: float, radius: float, num_events: int, filtered_dir: Path, knn_neighbors: int | None, connectivity: bool, num_cores: int):
    """Attach InfoNCE edges to one filtered_r*.pt file and save output."""

    suffix = int(recall * 1000)
    filtered_pt = filtered_dir / f"filtered_r{suffix}_data-{num_events}.pt"
    if not filtered_pt.exists():
        print(f"[skip] filtered dataset missing for recall {recall}: {filtered_pt}")
        return

    tracking_out = filtered_dir / f"tracking_filtered_r{suffix}_rad{radius:.2f}_data-{num_events}.pt"
    if tracking_out.exists():
        print(f"[skip] tracking dataset already exists: {tracking_out.name}")
        return

    print(f"  Building tracking dataset → {tracking_out.name}")
    add_edges_to_dataset(
        input_pt=filtered_pt,
        output_pt=tracking_out,
        radius=radius,
        k_max_neighbors=knn_neighbors,
        conn=connectivity,
        n_jobs=num_cores,
    )

def main():
    parser = argparse.ArgumentParser(description="Filter pileup dataset with trained model")
    parser.add_argument("-c","--ckpt-folder", default=CHECKPOINT_FOLDER, help="Subfolder in eval/pileup containing checkpoint")
    parser.add_argument("-pu","--pileup-density", default=PILEUP_DENSITY, help="Pileup density (50, 100, 200)")
    parser.add_argument("-n","--num-events", default=PILEUP_NUM_EVENTS, help="Number of events in the pileup dataset to filter, i.e. n --> 'data-n.pt'")
    parser.add_argument("-r","--recalls", nargs="+", type=float, default=RECALL_TARGETS, help="Recall targets e.g. -r 0.99 0.995 0.999")

    parser.add_argument("-nt", "--no-tracking", action="store_true", help="Only generate filtered pileup datasets and skip construction of tracking datasets with InfoNCE edges")
    parser.add_argument("-ot", "--tracking-only", action="store_true", help="Skip pileup filtering and build tracking dataset(s) from existing filtered files")
    parser.add_argument("-em", "--export-masks-only", action="store_true", help="Export event masks for each recall target and return.")

    parser.add_argument("-rad", "--graph-radius", nargs = "+", type=float, default=KNN_RADIUS, help="Radius hyperparameterto build radius_graphs in tracking task.")
    parser.add_argument("-knn", "--graph-k-neighbors", type=int, default=KNN_MAX_NEIGHBORS, help="Max number of neighbors for radius_graph in tracking task (if not specified, uses 1/5*sqrt(N)).")
    parser.add_argument("-conn","--connectivity", action="store_true", help="Calculate and save connectivity metrics (--> pileup/filtered_data/cluster_analysis/)")

    parser.add_argument("-ncpu", "--num-cores", type=int, default=CPU_CORES_FOR_MP, help="Parallel workers for edge generation")
    args = parser.parse_args()

    if not args.connectivity:
        args.connectivity = CONNECTIVITY_METRICS
    if not args.no_tracking:
        args.no_tracking = SKIP_TRACKING
    if not args.export_masks_only:
        args.export_masks_only = EXPORT_MASKS_ONLY

    ckpt_dir = root_path / f"eval/pileup/{args.ckpt_folder}"
    ckpt_path = ckpt_dir / "best.ckpt"
    hparams_path = ckpt_dir / "hparams.yaml"

    data_path = root_path / "data/processed/pileup" / f"pu{args.pileup_density}" / f"data-{args.num_events}.pt"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    filtered_data_dir = ckpt_dir / "filtered_data"
    filtered_data_dir.mkdir(exist_ok=True, parents=True)

    event_masks_dir = ckpt_dir / "event_masks"
    event_masks_dir.mkdir(exist_ok=True, parents=True)

    # if -ot, load in a filtered dataset and build tracking dataset directly
    if args.tracking_only:
        print("Building tracking datasets from existing filtered pileup data")
        for rad in args.graph_radius:
            for r in args.recalls:
                build_tracking_dataset(recall=r, radius=rad, num_events=args.num_events, filtered_dir=filtered_data_dir, knn_neighbors=args.graph_k_neighbors, connectivity=args.connectivity, num_cores=args.num_cores)
        print(f"\nTracking dataset(s) saved to: {filtered_data_dir}")
        return

    # otherwise, run the full pipeline
    data, slices, idx_split = torch.load(data_path, weights_only=False)
    dataset = EventDataset(data, slices)

    sample = dataset[0]
    in_dim = sample.x.shape[1]
    coords_dim = sample.coords.shape[1]
    model = _load_model(ckpt_path, hparams_path, in_dim=in_dim, coords_dim=coords_dim, task="pileup")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #torch.device("cpu")

    y_true_all: List[torch.Tensor] = []
    y_prob_all: List[torch.Tensor] = []
    preds_per_event: List[torch.Tensor] = []

    for evt in dataset:
        batch_evt = Batch.from_data_list([evt]).to(device)
        with torch.no_grad():
            logits = model(batch_evt).squeeze()
            probs = torch.sigmoid(logits).cpu()
        preds_per_event.append(probs)

        y_prob_all.append(probs)
        y_true_all.append(evt.y.cpu())

    y_true_cat = torch.cat(y_true_all)
    y_prob_cat = torch.cat(y_prob_all)

    recall_targets = args.recalls
    thresholds = {r: _threshold_at_recall(y_true_cat, y_prob_cat, r) for r in recall_targets}

    print(f"Recall targets: {recall_targets}")
    print("Thresholds:")
    for r, thr in thresholds.items():
        print(f"  r={r:.3f} --> thr={thr:.4f}")

    # Export masks for each recall target using the pre-computed threshold
    for r in recall_targets:
        export_masks(dataset, preds_per_event, r, event_masks_dir, args.num_events, threshold=thresholds[r])
    print(f"Event masks saved to: {event_masks_dir}\n")
    if args.export_masks_only:
        return

    pos_before = int((y_true_cat == 1).sum())
    neg_before = int((y_true_cat == 0).sum())
    ratio_before = 100 * pos_before / (pos_before + neg_before)
    print(
        f"Total points before filter - positives: {pos_before}, negatives: {neg_before} "
        f"({ratio_before:.2f}% true)\n"
    )
    filtered_files = []
    for r, thr in thresholds.items():
        filtered_events = [_filter_event(evt, p_evt, thr) for evt, p_evt in zip(dataset, preds_per_event)]

        data_filt, slices_filt = dataset.collate(filtered_events)

        y_after = torch.cat([evt.y for evt in filtered_events])
        pos_after = int((y_after == 1).sum())
        neg_after = int((y_after == 0).sum())
        ratio_after = 100 * pos_after / (pos_after + neg_after) if (pos_after + neg_after) > 0 else 0.0
        print(
            f"Recall ≥ {r:.1%} | positives: {pos_after}, negatives: {neg_after} "
            f"({ratio_after:.2f}% true)"
        )

        suffix = int(r * 1000)
        out_path_r = filtered_data_dir / f"filtered_r{suffix}_data-{args.num_events}.pt"
        torch.save((data_filt, slices_filt, idx_split), out_path_r)
        filtered_files.append((suffix, out_path_r))

    if not args.no_tracking and filtered_files:
        print("\nStarting construction of tracking datasets (InfoNCE edges)")
        for rad in args.graph_radius:
            for r in args.recalls:
                build_tracking_dataset(recall=r, radius=rad, num_events=args.num_events, filtered_dir=filtered_data_dir, knn_neighbors=args.graph_k_neighbors, connectivity=args.connectivity, num_cores=args.num_cores)

    print(f"\nAll datasets saved to: {filtered_data_dir}")

if __name__ == "__main__":
    main()

