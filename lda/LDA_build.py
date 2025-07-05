import sys
sys.path.append('..')
import argparse
from pathlib import Path
import torch
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import os
from tqdm import tqdm
from dataset import preprocess_data, FEATURE_NAMES

def run_lda(start_event_idx: int, end_event_idx: int, lda_comp: int = 7, log_pt: bool = True, pileup_density: str = "200"):
    """
    Runs LDA on true line segments aggregated across all point clouds in specified range.
    Saves the LDA model to use in dataset construction with dataset.py.
    """
    num_events = end_event_idx - start_event_idx
    raw_data_dir = Path(f"../data/raw/pu{pileup_density}")
    
    feats = []
    pids = []

    print(f"Processing events from {start_event_idx} to {end_event_idx - 1} in '{raw_data_dir}'...")
    for i in tqdm(range(start_event_idx, end_event_idx)):
        event_file_path = raw_data_dir / f"graph_{i}.pt"
        assert event_file_path.exists(), f"Event file {event_file_path} not found."

        raw_event_data = torch.load(event_file_path, map_location='cpu', weights_only=False)    
        data = preprocess_data(raw_event_data, evtid=i, phi_transform=True, pt_log=log_pt)

        feats.append(data.x)
        pids.append(data.particle_id)

    feats = torch.cat(feats, dim=0).numpy()
    pids = torch.cat(pids, dim=0).numpy()

    # Filter out LS fakes (particle_id == -1) before fitting LDA.
    mask = (pids != -1)
    feats = feats[mask]
    pids = pids[mask]

    unique_lbl = np.unique(pids)

    print(f"Fitting LDA with {feats.shape[0]} data points, {feats.shape[1]} features, and {len(unique_lbl)} unique classes, for {lda_comp} components...")
    
    lda = LinearDiscriminantAnalysis(n_components=lda_comp, solver='svd')
    lda.fit(feats, pids)

    scalings_matrix = lda.scalings_
    explained_variance_ratio = lda.explained_variance_ratio_
    total_explained_variance = np.sum(explained_variance_ratio)
    feat_names = FEATURE_NAMES

    model_data = {
        'feature_names': feat_names,
        'scalings_matrix': torch.from_numpy(scalings_matrix).float(),
        'explained_variance_ratio': torch.from_numpy(explained_variance_ratio).float()
    }

    output_path = Path(f"LDA_pu{pileup_density}{'' if log_pt else '_raw_pt'}.pt")

    torch.save(model_data, output_path)
    print(f"LDA model saved successfully to: {output_path}")
    if log_pt: print("  Using transformed pt values pt --> log(pt)")
    else: print("  Using raw pt values")
    print(f"  Components: {lda_comp}")
    print(f"  Total Explained Variance: {total_explained_variance:.4f}")
    print(f"  Explained Variance per Component: {explained_variance_ratio}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and save a standalone LDA model from raw event data.")
    parser.add_argument("-s", "--start_event_idx", type=int, default=0, help="Starting event index (inclusive).")
    parser.add_argument("-e", "--end_event_idx", type=int, default=100, help="Ending event index (exclusive).")
    parser.add_argument("-c", "--num_lda_components", type=int, default=7, help="Number of LDA components to calculate (default: 7).")
    parser.add_argument("-pt", "--log_pt", action="store_false", help="Use RAW pt values instead of log pt (default: log)")
    parser.add_argument("-pu", "--pileup_density", type=str, default="200", help="Pileup density (default: 200).")
    args = parser.parse_args()

    run_lda(args.start_event_idx, args.end_event_idx, args.num_lda_components, args.log_pt, args.pileup_density) 