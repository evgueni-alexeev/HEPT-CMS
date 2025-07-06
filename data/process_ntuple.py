import awkward as ak
import uproot
import argparse
from joblib import Parallel, delayed
from tqdm import tqdm
import torch
from torch_geometric.data import Data
import os

ROOT_DIR = "ntuple.root"
OUTPUT_DIR = "raw/test/"
START_EVENT = 0
END_EVENT = 10
CPU_CORES_FOR_MP = 8

def process_event(i, tree, track_label, cols, start_idx):
    event_data = tree.arrays(cols, entry_start = start_idx + i, entry_stop = start_idx + i + 1)[0]
    
    trks = torch.tensor([xx[0] for xx in event_data[track_label]])  # first if multiple tracks in LS (uncommon)
    
    md_idxA, md_idxE = ak.to_list(ak.to_dataframe(event_data[INDICES]).values.T)
    md_anchor0 = torch.tensor(ak.to_dataframe(event_data[MD0_F]).values[md_idxA])
    md_anchor1 = torch.tensor(ak.to_dataframe(event_data[MD0_F]).values[md_idxE])
    md_end0 = torch.tensor(ak.to_dataframe(event_data[MD1_F]).values[md_idxA])
    md_end1 = torch.tensor(ak.to_dataframe(event_data[MD1_F]).values[md_idxE])
    
    ls_features = torch.tensor(ak.to_dataframe(event_data[LS_F]).values)                   # LS: phi, eta, pt (phi --> sin/cos in dataset.py)
    ls_coords = 0.25*(md_anchor0 + md_end0 + md_anchor1 + md_end1)                         # LS (4 hit avg): r, x, y, z, dphi
    anchor_diff = (md_anchor1 - md_anchor0)                                                # Anchor MD: dr_A, dx_A, dy_A, dz_A, ddphi
    end_diff = (md_end1 - md_end0)                                                         # End MD:    dr_E, dx_E, dy_E, dz_E
    ls_diff = 0.5*((md_anchor1+md_anchor0)-(md_end1+md_end0))                              # MD diff:   dr, dx, dy, dz
    ls_anchor_layer = torch.tensor(ak.to_dataframe(event_data[MD_L_IDX]).values[md_idxA])  # Anchor MD layer
    ls_end_layer = torch.tensor(ak.to_dataframe(event_data[MD_L_IDX]).values[md_idxE])     # End MD layer
                                                                    
    data = torch.cat([ls_features,ls_coords,anchor_diff,end_diff[:,:-1],ls_diff[:,:-1],ls_anchor_layer,ls_end_layer],dim=1)
    
    graph = Data(x = data, particle_id = trks)
    output_file = os.path.join(args.out_dir, f"graph_{i+args.start_event}.pt")
    torch.save(graph, output_file)
    
    return graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data from root file")
    parser.add_argument("-s", "--start_event", type=int, default=START_EVENT)
    parser.add_argument("-e", "--end_event", type=int, default=END_EVENT)
    parser.add_argument("-o", "--out_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("-r", "--root_dir", type=str, default=ROOT_DIR)
    parser.add_argument("-c", "--cpu_cores", type=int, default=CPU_CORES_FOR_MP)
    args = parser.parse_args()

    num_events = args.end_event - args.start_event

    os.makedirs(args.out_dir, exist_ok=True)
    file = uproot.open(args.root_dir)
    tree = file["tree"]

    if "LS_SimTrkIdx" in tree.keys():
        track_label = "LS_SimTrkIdx"
    elif "LS_TCidx" in tree.keys():
        track_label = "LS_TCidx"
    else:
        raise ValueError("Neither of the truth label branches ('LS_SimTrkIdx' or 'LS_TCidx') found in the ROOT file.")

    LS_F = ["LS_phi", "LS_eta", "LS_pt"]
    MD0_F = ["MD_0_r", "MD_0_x", "MD_0_y", "MD_0_z", "MD_dphichange"]
    MD1_F = ["MD_1_r", "MD_1_x", "MD_1_y", "MD_1_z", "MD_dphichange"]
    INDICES = ["LS_MD_idx0","LS_MD_idx1"]
    MD_L_IDX = ["MD_layer"]
    LABELS = ["tc_lsIdx", track_label]   # track_label = LS_SimTrkIdx or LS_TCidx; tc_lsIdx is LST truth label
    CUTS = ["LS_isFake"]
    
    ALL_COLS = LS_F + MD0_F + MD1_F + INDICES + MD_L_IDX + LABELS + CUTS
    
    Parallel(n_jobs=args.cpu_cores)(
        delayed(process_event)(i, tree, track_label, ALL_COLS, args.start_event) for i in tqdm(range(num_events))
    )
    print(f"Saved graph_{args.start_event} to graph_{args.end_event - 1} from {args.root_dir} to {args.out_dir}")