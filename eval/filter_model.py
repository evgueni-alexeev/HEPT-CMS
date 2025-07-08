import sys
from pathlib import Path
from typing import List
import numpy as np
import argparse

root_path = Path(__file__).resolve().parents[1]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import torch
import yaml
from torch_geometric.data import Batch, InMemoryDataset
from sklearn.metrics import precision_recall_curve

from model import Transformer

CHECKPOINT_FOLDER = "pu200_0.409p"
PILEUP_DENSITY = 200
PILEUP_NUM_EVENTS = 5
RECALL_TARGETS = [0.985, 0.990, 0.995]          #[0.90, 0.95, 0.98, 0.99, 0.995, 0.999, 1.0]

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def main():
    parser = argparse.ArgumentParser(description="Filter pileup dataset with trained model")
    parser.add_argument("-c","--ckpt-folder", default=CHECKPOINT_FOLDER, help="Subfolder in eval/pileup containing checkpoint")
    parser.add_argument("-pu","--pileup-density", default=PILEUP_DENSITY, help="Pileup density (50, 100, 200)")
    parser.add_argument("-n","--num-events", default=PILEUP_NUM_EVENTS, help="Number of events in the pileup dataset to filter, i.e. n --> 'data-n.pt'")
    parser.add_argument("-r","--recalls", nargs="+", type=float, default=RECALL_TARGETS, help="Recall targets e.g. -r 0.99 0.995 0.999")
    args = parser.parse_args()

    ckpt_dir = root_path / f"eval/pileup/{args.ckpt_folder}"
    ckpt_path = ckpt_dir / "best.ckpt"
    hparams_path = ckpt_dir / "hparams.yaml"

    data_path = root_path / "data/processed/pileup" / f"pu{args.pileup_density}" / f"data-{args.num_events}.pt"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    filtered_data_dir = ckpt_dir / "filtered_data"
    filtered_data_dir.mkdir(exist_ok=True, parents=True)

    data, slices, idx_split = torch.load(data_path, weights_only=False)
    dataset = EventDataset(data, slices)

    sample = dataset[0]
    in_dim = sample.x.shape[1]
    coords_dim = sample.coords.shape[1]
    model = _load_model(ckpt_path, hparams_path, in_dim=in_dim, coords_dim=coords_dim, task="pileup")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    print(f"Thresholds: {[round(thr,4) for _, thr in thresholds.items()]}")
    pos_before = int((y_true_cat == 1).sum())
    neg_before = int((y_true_cat == 0).sum())
    ratio_before = 100 * pos_before / (pos_before + neg_before)
    print(
        f"Total points before filter - positives: {pos_before}, negatives: {neg_before} "
        f"({ratio_before:.2f}% true)\n"
    )
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
    print(f"\nFiltered datasets saved to: {filtered_data_dir}")

if __name__ == "__main__":
    main()

