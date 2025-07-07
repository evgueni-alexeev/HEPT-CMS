import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[1]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

import torch
import yaml
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, InMemoryDataset
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy as np
import os

from model import Transformer

root_dir = Path(__file__).parent / "pileup/pu200_0.409p"
ckpt_dir = root_dir / "best.ckpt"
yaml_dir = root_dir / "hparams.yaml"
test_dataset = Path("../data/processed/pileup/pu200/data-5.pt")

config = yaml.safe_load(yaml_dir.open("r").read())
PLOTS = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EventDataset(InMemoryDataset):
    def __init__(self, data, slices):
        super().__init__(root=None)
        self.data, self.slices = data, slices

ckpt = torch.load(ckpt_dir,map_location="cpu",weights_only=False)
clean_state = {k.replace("model.", "", 1): v
               for k, v in ckpt["state_dict"].items()
               if k.startswith("model.")}

data, slices, split_idx = torch.load(test_dataset,weights_only=False)
dataset = EventDataset(data,slices)
idx = split_idx['train']

test_loader = DataLoader(dataset.index_select(idx), batch_size=1, shuffle=False)

m = Transformer(in_dim=test_loader.dataset[0].x.shape[1], coords_dim=test_loader.dataset[0].coords.shape[1], task='pileup',**config["model_kwargs"])
m.load_state_dict(clean_state, strict=True)
m = m.to(device)

y_true = torch.tensor([],device=device)
y_embs = torch.tensor([],device=device)
y_true1 = torch.tensor([],device=device)
y_embs1 = torch.tensor([],device=device)
y_true2 = torch.tensor([],device=device)
y_embs2 = torch.tensor([],device=device)

individual_prcs = []      # All data
individual_prcs1 = []     # pt < 1.5
individual_prcs2 = []     # pt > 1.5

for i in range(len(dataset)):
    temp_true = torch.tensor([],device=device)
    temp_embs = torch.tensor([],device=device)
    
    with torch.no_grad():
        m.eval()
        evnum=i
        data = Batch.from_data_list([dataset[i]]).to(device)
        emb = m(data)
        temp_embs = torch.cat((temp_embs,emb.squeeze()))
        temp_true = torch.cat((temp_true,data.y)).long()
    
    mask1 = (data.x[:,3]<0.47) #& (data.x[:,2] > -2.5) & (data.x[:,2] < 2.5)
    mask2 = (data.x[:,3]>0.47) #& (data.x[:,2] > -2.5) & (data.x[:,2] < 2.5)
    
    # Calculate individual event PRC curves
    if len(torch.unique(temp_true)) > 1:  # Only if we have both classes
        prec, rec, _ = precision_recall_curve(temp_true.cpu(), temp_embs.cpu())
        individual_prcs.append((rec, prec, evnum))
    
    if len(torch.unique(temp_true[mask1])) > 1 and mask1.sum() > 0:
        prec1, rec1, _ = precision_recall_curve(temp_true[mask1].cpu(), temp_embs[mask1].cpu())
        individual_prcs1.append((rec1, prec1, evnum))
    
    if len(torch.unique(temp_true[mask2])) > 1 and mask2.sum() > 0:
        prec2, rec2, _ = precision_recall_curve(temp_true[mask2].cpu(), temp_embs[mask2].cpu())
        individual_prcs2.append((rec2, prec2, evnum))
    
    # Aggregate data
    y_true = torch.cat((y_true,temp_true))
    y_embs = torch.cat((y_embs,temp_embs))
    y_true1 = torch.cat((y_true1,temp_true[mask1]))
    y_embs1 = torch.cat((y_embs1,temp_embs[mask1]))
    y_true2 = torch.cat((y_true2,temp_true[mask2]))
    y_embs2 = torch.cat((y_embs2,temp_embs[mask2]))
    
    print(evnum)

# Calculate aggregate PRC curves
plt_prec,plt_rec,_ = precision_recall_curve(y_true.cpu(), y_embs.cpu())     
plt_prec1,plt_rec1,_ = precision_recall_curve(y_true1.cpu(), y_embs1.cpu())    
plt_prec2,plt_rec2,_ = precision_recall_curve(y_true2.cpu(), y_embs2.cpu())    

if PLOTS:
    # Create three separate plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25.6, 14.4))

    # Plot 1: All data (individual + aggregate)
    for rec, prec, evnum in individual_prcs:
        ax1.plot(rec, prec, 'b-', alpha=0.3, linewidth=0.8)
    ax1.plot(plt_rec, plt_prec, 'b-', linewidth=3, label=f'Aggregate - All LS ({len(y_true):.1E} points)')
    ax1.axvline(x=0.99, color='red', linestyle='--', linewidth=1)
    ax1.set_xlabel('Recall (TP/TP+FN)')
    ax1.set_ylabel('Precision (TP/TP+FP)')
    ax1.set_ylim(0.0, 1.0)
    ax1.set_xlim(0.9, 1.0)
    ax1.set_title(f'PRC - All Data\n({len(individual_prcs)} individual events)')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray', alpha=0.3)
    ax1.minorticks_on()
    ax1.legend()

    # Plot 2: pt < 1.5 (individual + aggregate)
    for rec, prec, evnum in individual_prcs1:
        ax2.plot(rec, prec, 'g-', alpha=0.3, linewidth=0.8)
    ax2.plot(plt_rec1, plt_prec1, 'g-', linewidth=3, label=f'Aggregate - pt < 0.9 ({len(y_true1):.1E} points)')
    ax2.axvline(x=0.99, color='red', linestyle='--', linewidth=1)
    ax2.set_xlabel('Recall (TP/TP+FN)')
    ax2.set_ylabel('Precision (TP/TP+FP)')
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlim(0.9, 1.0)
    ax2.set_title(f'PRC - Low pT (pt < 0.9)\n({len(individual_prcs1)} individual events)')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray', alpha=0.3)
    ax2.minorticks_on()
    ax2.legend()

    # Plot 3: pt > 1.5 (individual + aggregate)
    for rec, prec, evnum in individual_prcs2:
        ax3.plot(rec, prec, 'r-', alpha=0.3, linewidth=0.8)
    ax3.plot(plt_rec2, plt_prec2, 'r-', linewidth=3, label=f'Aggregate - pt > 0.9 ({len(y_true2):.1E} points)')
    ax3.axvline(x=0.99, color='red', linestyle='--', linewidth=1)
    ax3.set_xlabel('Recall (TP/TP+FN)')
    ax3.set_ylabel('Precision (TP/TP+FP)')
    ax3.set_ylim(0.0, 1.0)
    ax3.set_xlim(0.9, 1.0)
    ax3.set_title(f'PRC - High pT (pt > 0.9)\n({len(individual_prcs2)} individual events)')
    ax3.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray', alpha=0.3)
    ax3.minorticks_on()
    ax3.legend()

    plt.tight_layout()
    # Save the three-panel plot
    plot_dir = root_dir
    plt.savefig(os.path.join(plot_dir, "prc_individuals.pdf"))
    plt.show()

    # Create the original combined plot
    plt.figure(figsize=(25.6, 14.4))
    plt.plot(plt_rec, plt_prec, linestyle="-", linewidth=2, color='blue', label='All LS')
    plt.plot(plt_rec1, plt_prec1, linestyle="-", linewidth=2, color='green', label=f'pt < 0.9 ({len(y_true1):.1E})')
    plt.plot(plt_rec2, plt_prec2, linestyle="-", linewidth=2, color='red', label=f'pt > 0.9 ({len(y_true2):.1E})')
    plt.axvline(x=0.99, color='red', linestyle='-', linewidth=1)

    # Add intersection points to the non-zoomed plot
    thresholds = [0.985, 0.99, 0.995]
    curve_data = [
        (plt_rec, plt_prec, 'All LS', 'blue'),
        (plt_rec1, plt_prec1, 'pt < 0.9', 'green'),
        (plt_rec2, plt_prec2, 'pt > 0.9', 'red')
    ]
    for threshold in thresholds:
        if threshold == 0.99:
            for (rec, prec, label, color) in curve_data:
                idx = np.abs(rec - threshold).argmin()
                plt.plot(rec[idx], prec[idx], 'o', markersize=12, color=color, markeredgecolor='black')

    plt.xlabel('Recall (TP/TP+FN)')
    plt.ylabel('Precision (TP/TP+FP)')
    plt.ylim(0.0, 1.0)
    plt.xlim(0.9, 1.0)
    plt.title('PRC Aggregate Comparison (trained for 400 epochs, 680 events)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray', alpha=0.3)
    plt.minorticks_on()
    plt.legend()
    # Save the non-zoomed aggregate plot
    plt.savefig(os.path.join(plot_dir, "prc_aggregate.pdf"))
    plt.show()

    # Create zoomed plot with intersection points
    plt.figure(figsize=(25.6, 14.4))
    plt.plot(plt_rec, plt_prec, linestyle="-", linewidth=2, color='blue', label='All LS')
    plt.plot(plt_rec1, plt_prec1, linestyle="-", linewidth=2, color='green', label=f'pt < 0.9 ({len(y_true1):.1E})')
    plt.plot(plt_rec2, plt_prec2, linestyle="-", linewidth=2, color='red', label=f'pt > 0.9 ({len(y_true2):.1E})')

    # Add vertical lines and intersection points
    for threshold in thresholds:
        plt.axvline(x=threshold, color='red', linestyle='--', linewidth=1)
        for (rec, prec, label, color) in curve_data:
            idx = np.abs(rec - threshold).argmin()
            plt.plot(rec[idx], prec[idx], 'o', markersize=12, color=color, markeredgecolor='black')

    plt.xlabel('Recall (TP/TP+FN)')
    plt.ylabel('Precision (TP/TP+FP)')
    plt.ylim(0.0, 0.6)
    plt.xlim(0.98, 1.0)
    plt.title('PRC Aggregate Comparison (Zoomed)')
    plt.grid(True, which='both', linestyle='--', linewidth=0.3, color='gray', alpha=0.3)
    plt.minorticks_on()
    legend = plt.legend(loc='upper center',bbox_to_anchor=(0.825,0.995))

    # Prepare formatted precision values text (all-caps recall, aligned columns)
    recall_labels = [f"RECALL = {t:.3f}" for t in thresholds]
    lines = []
    for i, threshold in enumerate(thresholds):
        lines.append(f"{recall_labels[i]}")
        for (rec, prec, label, color) in curve_data:
            idx = np.abs(rec - threshold).argmin()
            lines.append(f"{label:<10}: {prec[idx]:.3f}")
        lines.append("")
    box_text = "\n".join(lines)

    # Place the box under the legend, 20% smaller font
    fontsize = 13
    at = AnchoredText(box_text, loc='upper right', prop={'family': 'monospace', 'size': fontsize}, frameon=True, pad=0.7, borderpad=1.0)
    at.patch.set_boxstyle("round,pad=0.5")
    at.patch.set_facecolor('white')
    at.patch.set_edgecolor('black')
    at.patch.set_linewidth(1.5)
    plt.gca().add_artist(at)

    # Save the zoomed aggregate plot
    plt.savefig(os.path.join(plot_dir, "prc_aggregate_zoomed.pdf"))
    plt.show()

torch.cuda.empty_cache()