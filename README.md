# HEPT-CMS
Implementation of HEPT efficient transformer model utilizing LSH (https://arxiv.org/abs/2402.12535) for high pileup CMS data. Transformer/efficient attention and hashing code was adapted from Siqi Miao's repo: https://github.com/Graph-COM/HEPT, who is one of the authors on the paper.

Currently set up to run tracking and pileup tasks using **LST line segment (LS)** features.
- **Pileup** model does binary classification on LS using `LS_isFake` branch as labels
- **Tracking** model outputs 24D learned embedding space where true LS from the same sim track (`LS_SimTrkIdx` branch) are clustered together, with no additional post-processing

### Dataset construction

Raw event data is generated from pileup 50, 100 or 200 ntuple (not included in repo due to size) via `data/process_ntuple.py` and saved as `graph_#.pt` files. There are a few raw events included in the repo for illustrative purposes (not enough for proper training). To generate the combined dataset from those event files, run
```
python dataset.py -d <task> -pu <pileup density> ...
```
with `<task> = pileup` or `tracking`, and `<pileup density> = {50,100,200}`. Use `-t <N>` command to only use first N events in the correspoding raw folder (will generate dataset with all events in raw by default).

Tracking dataset construction is the same as pileup, with additional LS-LS edges built for contrastive learning (used during training only). Edges are built using `torch_geometric.nn.radius_graph` in LDA-transformed 3D coordinate space which cluster the line segments closer together if they are from the same track. Using LDA coords instead of e.g. $\eta$ and $\phi$ massively reduces the radius and k hyperparameters required to saturate each sim track cluster.

### Training

Model training and inference is done with `PyTorch Lightning`, supporting both cpu, and single/multi-gpu (via `FSDP`) setups. Training settings and model hyperparameters are controlled through `cfg_pileup.yaml` and `cfg_tracking.yaml`, which are essentially identical. To train the model, run
```
python trainer.py -p
```
or
```
python trainer.py -t
```
for pileup and tracking respectively.

For pileup, Focal Loss (https://paperswithcode.com/method/focal-loss) is used, with emphasis on maximizing precision at high recall (0.99 to 0.999). The pileup dataset (especially pu200) is highly imbalanced, and we find better performance using both $\gamma \sim 2$ and an overtuned (2x) imbalance ratio in the loss function, calculated separately for each event, which helps prioritize the true LS and focus on high recall.

For tracking, InfoNCE (https://arxiv.org/pdf/1807.03748.pdf) is used on the LS-LS edges, which currently only uses edges between true LS. An edge connecting a LS to another LS in the same sim track is classified as positive, and contrasted with the other N-1 negative edges from that LS. <u>NOTE</u>: pileup/fake LS are currently excluded from training, incorporating them into the loss function is a work in progress. However, once trained the model can be applied to e.g. a pileup dataset that was partially filtered using the pileup model.

### Evaluation

Logs (metrics, plots) from the training run will be saved in `logs/<task>/csv/version_XX/`, with optional tensorboard output in `tb` folder. The model also saves the best and last checkpoints during training.

- Pileup outputs include F-score ($F_1$ and $F_\beta$, usually with $\beta = 3$), the Precision-Recall Curve and AUPRC
- Tracking outputs the (mean) Average Precision @ k: AP@k = $\frac{1}{N}\sum_{i=1}^{N}$ Prec @ $k_i$ i.e. for each LS, find the sim track index and check how many of its k = 1, 2, ..., L-1 nearest neighbors (L = track length) in the latent space belong to the same sim track, and average across k and across all LS. Note this is ideologically similar to efficiency, but goes point by point rather than track by track. AP@k is calculated for 3 thresholds: $p_t \geq \{0, 0.5, 0,9\}$

There is some post-processing done after the training loop to generate the plots and fix the output files `metrics.csv`. For early stopping during training, if those outputs are needed/important, create a blank "STOP" file in the root dir (e.g. `touch STOP' in bash) and the program will exit the training loop gracefully and still generate plots/post-processing.

`eval/filter_model.py` can be used to apply a trained pileup model on a dataset to produce filtered versions at various recall thresholds, those will be saved in `eval/pileup/<saved_model>/filtered_data`. The filtered(or unfiltered) datasets can also be analyzed using a trained tracking model via `eval/tracking_eval.py`. This gives a detailed breakdown of efficiency and analysis by track length, $p_t$, etc when applied on a partially cleaned dataset. The same analysis can be run on just the sim-matched line segments in the dataset (which masks out `LS_isFake` points).

There are currently 2 pre-trained models saved in `eval`, both trained on 2000 events from the pu200 ntuple, on a single A40 over ~200 epochs. The pileup model achieved a 40.9% precision at 99% recall, and 96.5% AUPRC, and the tracking model achieved a mean AP score of 99.1%, trained/evaluated just the sim-matched LS.