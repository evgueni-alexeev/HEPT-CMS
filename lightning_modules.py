import lightning as L
import numpy as np
import pandas as pd
import os
import torch
import warnings
from pathlib import Path
from contextlib import nullcontext
from torch_geometric.loader import DataLoader
import torch.distributed as dist
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from torchmetrics import MeanMetric
from torchmetrics.classification import BinaryAveragePrecision, BinaryF1Score, BinaryPrecisionAtFixedRecall, BinaryRecallAtFixedPrecision,BinaryFBetaScore
from lightning.pytorch.callbacks import Callback
from lightning_fabric.utilities.rank_zero import rank_zero_only
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch_geometric.utils import unbatch
from fvcore.nn import FlopCountAnalysis, flop_count_table

from utils import get_optimizer, get_lr_scheduler, get_loss, calc_AP_at_k, point_filter
from model import Transformer
from dataset import TrackingPileup, TrackingPileupTransform

def get_dataset(task, num_events, pileup_density):
    dataset = TrackingPileup(transform=TrackingPileupTransform(), truncate=None, num_events = num_events, task = task, pileup_density = pileup_density)
    return dataset

#============================================
#          LIGHTNING MODULES
#============================================

class TrackLightning(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = None           # model init in setup()
        self.criterion = get_loss(config["loss_name"], config["loss_kwargs"])
        self.task = self.hparams.task

        # Pileup metrics - AUC-PR, F1, FB
        if self.task == "pileup":
            self.pileup_metrics = torch.nn.ModuleDict({
                f"{phase}_auc_pr": BinaryAveragePrecision()
                for phase in ["train", "val", "test"]
            })
            self.pileup_metrics.update({
                f"{phase}_f1": BinaryF1Score()
                for phase in ["train", "val", "test"]
            })
            self.pileup_metrics.update({
                f"{phase}_fb": BinaryFBetaScore(beta=config["f_beta"])
                for phase in ["train", "val", "test"]
            })
        # Tracking metrics - Average Precision@k (AP@k) = 1/n sum_k^n(precision@k)
        elif self.task == "tracking":
            self.pt_thres = config.get("pt_thres", [0, 0.5, 0.9])
            metric_names = ["AP"]
            self.tracking_metrics = torch.nn.ModuleDict({
                f"{phase}_{name}_{str(pt).replace('.', 'p')}" : MeanMetric(nan_strategy="error")
                for name in metric_names for pt in self.pt_thres for phase in ["train", "val", "test"]
            })
            self.tracking_metrics["loss"] = MeanMetric(nan_strategy="error")
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def setup(self, stage=None):
        if self.model is None:
            dm = self.trainer.datamodule
            if not hasattr(dm, 'ds') or dm.ds is None:
                 dm.setup(stage='fit')
            self.model = Transformer(in_dim = dm.ds.x_dim, coords_dim = dm.ds.coords_dim, task = dm.ds.task, **self.hparams["model_kwargs"])
            _log_model_info(self.model, dm.ds, 120000, 1)  # calc flops/params based on ~avg event size

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        if self.task == "pileup":
            yhat = self(batch).squeeze()
            imbalance_ratio = getattr(batch, 'imb', None)
            if imbalance_ratio is not None and imbalance_ratio.numel() > 1:
                imbalance_ratio = imbalance_ratio.mean()
            elif imbalance_ratio is not None:
                imbalance_ratio = imbalance_ratio.squeeze()
            loss = self.criterion(yhat, batch.y.float(), imbalance_ratio)
            bs = batch.y.float().size(0)
            with torch.no_grad():
                probs = torch.sigmoid(yhat)
                preds = (probs > 0.5).int()
                ys = batch.y.int()

            self.pileup_metrics["train_auc_pr"].update(probs, ys)
            self.pileup_metrics["train_f1"].update(preds, ys)
            self.pileup_metrics["train_fb"].update(preds, ys)

            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
            return loss
        elif self.task == "tracking":
            embeddings = self(batch)
            tracklen = getattr(batch, 'tracklen', None)
            loss = self.criterion(embeddings, batch.point_pairs_index, batch.particle_id, batch.reconstructable, batch.pt, tracklen=tracklen)
            bs = batch.batch.max().item() + 1

            outputs = {
                "loss": loss.detach(),
                "embeddings": embeddings.detach(),
                "particle_id": batch.particle_id.detach(),
                "reconstructable": batch.reconstructable.detach(),
                "pt": batch.pt.detach(),
                "batch": batch.batch.detach(),
                "tracklen": batch.tracklen.detach()
            }
            self._update_tracking_metrics(outputs, "train")
            self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
        return loss

    def on_train_epoch_end(self):
        if self.task == "pileup":
            # Compute and log metrics at epoch end
            self.log_dict({
                'train_f1': self.pileup_metrics["train_f1"].compute(),
                'train_auc_pr': self.pileup_metrics["train_auc_pr"].compute(),
                'train_fb': self.pileup_metrics["train_fb"].compute()
            }, prog_bar=False, logger=True, sync_dist=True)
            # Reset metrics after logging
            self.pileup_metrics["train_auc_pr"].reset()
            self.pileup_metrics["train_f1"].reset()
            self.pileup_metrics["train_fb"].reset()
        elif self.task == "tracking":
            self._compute_and_log_tracking_metrics("train")
            self._reset_tracking_metrics("train")

    def validation_step(self, batch, batch_idx):
        if self.task == "pileup":
            yhat = self(batch).squeeze()
            imbalance_ratio = getattr(batch, 'imb', None)
            if imbalance_ratio is not None and imbalance_ratio.numel() > 1:
                imbalance_ratio = imbalance_ratio.mean()
            elif imbalance_ratio is not None:
                imbalance_ratio = imbalance_ratio.squeeze()
            loss = self.criterion(yhat, batch.y.float(), imbalance_ratio)
            bs = batch.y.float().size(0)
            with torch.no_grad():
                probs = torch.sigmoid(yhat)
                preds = (probs > 0.5).int()
                ys = batch.y.int()
            outputs = {"logits": yhat.detach(), "y": batch.y.detach(), "loss": loss.detach()}
            # Only update metrics, don't compute or log them here
            self.pileup_metrics["val_auc_pr"].update(probs, ys)
            self.pileup_metrics["val_f1"].update(preds, ys)
            self.pileup_metrics["val_fb"].update(preds, ys)
            # Only log loss during validation steps
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
            return outputs
        elif self.task == "tracking":
            embeddings = self(batch)
            tracklen = getattr(batch, 'tracklen', None)
            loss = self.criterion(embeddings, batch.point_pairs_index, batch.particle_id, batch.reconstructable, batch.pt, tracklen=tracklen)
            bs = batch.batch.max().item() + 1

            outputs = {
                "loss": loss.detach(),
                "embeddings": embeddings.detach(),
                "particle_id": batch.particle_id.detach(),
                "reconstructable": batch.reconstructable.detach(),
                "pt": batch.pt.detach(),
                "batch": batch.batch.detach(),
                "tracklen": batch.tracklen.detach()
            }
            self._update_tracking_metrics(outputs, "val")
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
            return outputs

    def on_validation_epoch_end(self):
        if self.task == "pileup":
            # Compute and log metrics at epoch end
            self.log_dict({
                'val_f1': self.pileup_metrics["val_f1"].compute(),
                'val_auc_pr': self.pileup_metrics["val_auc_pr"].compute(),
                'val_fb': self.pileup_metrics["val_fb"].compute()
            }, prog_bar=True, logger=True, sync_dist=True)
            # Reset metrics after logging
            self.pileup_metrics["val_auc_pr"].reset()
            self.pileup_metrics["val_f1"].reset()
            self.pileup_metrics["val_fb"].reset()
        elif self.task == "tracking":
            self._compute_and_log_tracking_metrics("val")
            self._reset_tracking_metrics("val")

    def test_step(self, batch, batch_idx):
        if self.task == "pileup":
            yhat = self(batch).squeeze()
            imbalance_ratio = getattr(batch, 'imb', None)
            if imbalance_ratio is not None and imbalance_ratio.numel() > 1:
                imbalance_ratio = imbalance_ratio.mean()
            elif imbalance_ratio is not None:
                imbalance_ratio = imbalance_ratio.squeeze()
            loss = self.criterion(yhat, batch.y.float(), imbalance_ratio)
            bs = batch.y.float().size(0)
            with torch.no_grad():
                probs = torch.sigmoid(yhat)
                preds = (probs > 0.5).int()
                ys = batch.y.int()
            # Only update metrics, don't compute or log them here
            self.pileup_metrics["test_auc_pr"].update(probs, ys)
            self.pileup_metrics["test_f1"].update(preds, ys)
            self.pileup_metrics["test_fb"].update(preds, ys)
            # Only log loss during test steps
            self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
            return {"loss": loss}
        elif self.task == "tracking":
            embeddings = self(batch)
            tracklen = getattr(batch, 'tracklen', None)
            loss = self.criterion(embeddings, batch.point_pairs_index, batch.particle_id, batch.reconstructable, batch.pt, tracklen=tracklen)
            bs = batch.batch.max().item() + 1

            outputs = {
                "loss": loss.detach(),
                "embeddings": embeddings.detach(),
                "particle_id": batch.particle_id.detach(),
                "reconstructable": batch.reconstructable.detach(),
                "pt": batch.pt.detach(),
                "batch": batch.batch.detach(),
                "tracklen": batch.tracklen.detach()
            }
            self._update_tracking_metrics(outputs, "test")
            self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=bs)
            return outputs

    def on_test_epoch_end(self):
        if self.task == "pileup":
            # Compute and log metrics at epoch end
            self.log_dict({
                'test_f1': self.pileup_metrics["test_f1"].compute(),
                'test_auc_pr': self.pileup_metrics["test_auc_pr"].compute(),
                'test_fb': self.pileup_metrics["test_fb"].compute()
            }, prog_bar=True, logger=True, sync_dist=True)
            # Reset metrics after logging
            self.pileup_metrics["test_auc_pr"].reset()
            self.pileup_metrics["test_f1"].reset()
            self.pileup_metrics["test_fb"].reset()
        elif self.task == "tracking":
            self._compute_and_log_tracking_metrics("test")
            self._reset_tracking_metrics("test")

    def _update_tracking_metrics(self, outputs, phase):
        self.tracking_metrics["loss"].update(outputs["loss"])
        
        embeddings_list = unbatch(outputs["embeddings"], outputs["batch"])
        cluster_ids_list = unbatch(outputs["particle_id"], outputs["batch"])
        tracklen_list = unbatch(outputs["tracklen"], outputs["batch"])
        reconstructable_list = unbatch(outputs["reconstructable"], outputs["batch"])
        pt_list = unbatch(outputs["pt"], outputs["batch"])
        batch_idx_list = unbatch(outputs["batch"], outputs["batch"])

        for pt_thres in self.pt_thres:
            batch_mask = point_filter(outputs["particle_id"], outputs["reconstructable"], outputs["pt"], pt_thres=pt_thres)
            mask_list = unbatch(batch_mask, outputs["batch"])
            
            for i in range(len(embeddings_list)):
                if len(embeddings_list[i]) > 1:
                    key_suffix = str(pt_thres).replace('.', 'p')
                    ap_at_k = calc_AP_at_k(
                        embeddings_list[i], cluster_ids_list[i], tracklen_list[i], mask_list[i],
                        self.hparams.loss_kwargs.get('dist_metric', 'cosine')
                    )
                    self.tracking_metrics[f"{phase}_AP_{key_suffix}"].update(ap_at_k)

    def _compute_and_log_tracking_metrics(self, phase):
        # Dictionary for metrics that are NOT the main_metric, to be logged via self.log_dict
        other_metrics_to_log = {}
        main_metric_name = self.hparams.main_metric

        for pt_thres in self.pt_thres:
            key_suffix = str(pt_thres).replace('.', 'p')
            module_dict_key = f"{phase}_AP_{key_suffix}"
            log_key = f"{phase}_AP@{pt_thres}"
            metric_val = self.tracking_metrics[module_dict_key].compute().item()
                                
            if log_key == main_metric_name:
                self.log(main_metric_name, metric_val, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
            else:
                self.log(log_key, metric_val, prog_bar=False, on_epoch=True, sync_dist=True)

    def _reset_tracking_metrics(self, phase=None):
        if phase is None:
            # Reset all metrics
            for metric in self.tracking_metrics.values():
                metric.reset()
        else:
            # Only reset metrics for the given phase
            for key, metric in self.tracking_metrics.items():
                if key.startswith(f"{phase}_") or key == "loss":
                    metric.reset()

    def configure_optimizers(self):
        optimizer = get_optimizer(self.model.parameters(), self.hparams["optimizer_name"], self.hparams["optimizer_kwargs"])
        lr_metric = self.hparams.get("lr_scheduler_metric", "val_loss")

        lr_s = get_lr_scheduler(optimizer, self.hparams["lr_scheduler_name"], self.hparams["lr_scheduler_kwargs"])
        if lr_s is None:
            return optimizer
        else:
            interval = "step" if self.hparams["lr_scheduler_name"] == "cosine" else "epoch"
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_s,
                    "monitor": lr_metric,
                    "interval": interval,
                },
            }

class  TrackDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.task = config["task"]
        self.pu = config["pileup_density"]
        self.ds = None
        if isinstance(self.config["num_events"],int):
            self.evts = self.config["num_events"]
        else:
            self.evts = 120

    def setup(self, stage=None):
        if self.ds is None:
            if self.task == "pileup" or self.task == "tracking":
                self.ds = get_dataset(num_events=self.evts, task=self.task, pileup_density=self.pu)
            else:
                raise ValueError(f"Unknown task for dataset loading: {self.task}")

    def train_dataloader(self):
        return DataLoader(
            self.ds[self.ds.idx_split["train"]],
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers = self.config.get("num_threads", 0),
            pin_memory=True,
            persistent_workers = True if self.config.get("num_threads", 0) > 0 else False,
            prefetch_factor=4 if self.config.get("num_threads", 0) > 0 else None
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds[self.ds.idx_split["valid"]],
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers = self.config.get("num_threads", 0),
            pin_memory=True,
            persistent_workers = True if self.config.get("num_threads", 0) > 0 else False,
            prefetch_factor=4 if self.config.get("num_threads", 0) > 0 else None
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds[self.ds.idx_split["test"]],
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers = self.config.get("num_threads", 0),
            pin_memory=True,
            persistent_workers = True if self.config.get("num_threads", 0) > 0 else False,
            prefetch_factor=4 if self.config.get("num_threads", 0) > 0 else None
        )

#============================================
#                 CALLBACKS
#============================================

class VRAMTracker(Callback):
    """
    Track GPU memory usage during training.
    """
    def on_validation_epoch_end(self, trainer, pl_module):
        if not torch.cuda.is_available(): return
        res = torch.cuda.max_memory_reserved() // 10**6
        local_t = torch.tensor(res, device=pl_module.device)
        if not dist.is_available() or not dist.is_initialized():
            if pl_module.global_rank == 0:
                print(f" VRAM usage max peak: {int(local_t.item())} MB")
            return

        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(local_t) for _ in range(world_size)]
        dist.all_gather(gathered, local_t)
        if dist.get_rank() == 0:
            peaks = [int(x.item()) for x in gathered]
            avg = int(np.mean(peaks))
            max_val = int(np.max(peaks))
            print(f" VRAM usage avg: {avg} | max: {max_val} | all: {peaks}")

class PlotMetrics(Callback):
    """
    Save/write to file and generate plots for training/validation metrics.
    Calculate precision-recall curve and p@99r/r@95p for pileup.
    Postprocessing of metrics.csv file.
    """
    def __init__(self):
        super().__init__()
        self.history = {}
        self.preds, self.trues = [], []
        self.test_preds, self.test_trues = [], []

    def _init_history(self, pl_module):
        self.history = {}
        task = pl_module.hparams.task
        if task == "pileup":
            keys = ["train_loss", "val_loss", "train_f1", "val_f1", "train_fb", "val_fb", "train_auc_pr", "val_auc_pr"]
        elif task == "tracking":
            keys = ["train_loss", "val_loss"]
            for pt in pl_module.pt_thres:
                 keys.extend([f"train_AP@{pt}", f"val_AP@{pt}"])
        else: return

        for key in keys: self.history[key] = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if pl_module.hparams.task == "pileup" and outputs is not None and "logits" in outputs:
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            self.preds.append(probs)
            self.trues.append(outputs["y"].cpu().numpy())

        if pl_module.hparams.task == "pileup" and outputs is not None and "logits" in outputs:
            probs = torch.sigmoid(outputs["logits"]).cpu().numpy()
            self.test_preds.append(probs)
            self.test_trues.append(outputs["y"].cpu().numpy())

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self.history: self._init_history(pl_module)

        metrics = trainer.callback_metrics
        task = pl_module.hparams.task

        if task == "pileup":
            self.history["val_loss"].append(metrics.get("val_loss", torch.tensor(float('nan'))).item())
            self.history["val_f1"].append(metrics.get("val_f1", torch.tensor(float('nan'))).item())
            self.history["val_fb"].append(metrics.get("val_fb", torch.tensor(float('nan'))).item())
            self.history["val_auc_pr"].append(metrics.get("val_auc_pr", torch.tensor(float('nan'))).item())
        elif task == "tracking":
             self.history["val_loss"].append(metrics["val_loss"].item())
             for pt in pl_module.pt_thres:
                 self.history[f"val_AP@{pt}"].append(metrics.get(f"val_AP@{pt}", torch.tensor(float('nan'))).item())

    def on_train_epoch_start(self, trainer, pl_module):
        if pl_module.hparams.task == "pileup":
            self.preds.clear()
            self.trues.clear()

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.history: self._init_history(pl_module)

        metrics = trainer.callback_metrics
        task = pl_module.hparams.task

        if task == "pileup":
            self.history["train_loss"].append(metrics.get("train_loss", torch.tensor(float('nan'))).item())
            self.history["train_f1"].append(metrics.get("train_f1", torch.tensor(float('nan'))).item())
            self.history["train_fb"].append(metrics.get("train_fb", torch.tensor(float('nan'))).item())
            self.history["train_auc_pr"].append(metrics.get("train_auc_pr", torch.tensor(float('nan'))).item())
        elif task == "tracking":
             self.history["train_loss"].append(metrics["train_loss"].item())
             for pt in pl_module.pt_thres:
                 self.history[f"train_AP@{pt}"].append(metrics.get(f"train_AP@{pt}", torch.tensor(float('nan'))).item())

    @rank_zero_only
    def on_fit_end(self, trainer, pl_module):
        if not self.history or not self.history["train_loss"]:
            print("PlotMetrics: No metric history found, skipping plotting.")
            return

        out_dir = Path(f"{trainer.logger.log_dir}/plots")
        os.makedirs(out_dir, exist_ok=True)
        epochs = np.arange(1, len(self.history["train_loss"]) + 1)
        task = pl_module.hparams.task

        # 1) Loss plot
        fig1 = plt.figure(figsize=(16,9))
        plt.plot( epochs, self.history["train_loss"], label="train loss")
        plt.plot( epochs, self.history["val_loss"], label="val loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch")
        plt.legend(); plt.tight_layout()
        fig1.savefig(f"{out_dir}/loss.pdf"); plt.close(fig1)

        df = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

        lr_cols = [c for c in df.columns if c.startswith("lr-")]
        if lr_cols:
            df[lr_cols] = df[lr_cols].ffill()

        df = df.groupby(["epoch", "step"], as_index=False).first()
        df.to_csv(f"{trainer.logger.log_dir}/metrics.csv", index=False)

        if task == "pileup":
            beta = pl_module.hparams.get("f_beta", 1.0)
            
            # F1 score plot
            fig2 = plt.figure(figsize=(16,9))
            plt.plot(epochs, self.history["train_f1"], label="train F1")
            plt.plot(epochs, self.history["val_f1"], label="val F1")
            plt.xlabel("Epoch"); plt.ylabel("F1 Score"); plt.title("F1 Score vs Epoch")
            plt.legend(); plt.tight_layout()
            fig2.savefig(f"{out_dir}/F1_Score.pdf"); plt.close(fig2)

            # F-beta score plot
            fig3 = plt.figure(figsize=(16,9))
            plt.plot(epochs, self.history["train_fb"], label=f"train F{beta}")
            plt.plot(epochs, self.history["val_fb"], label=f"val F{beta}")
            plt.xlabel("Epoch"); plt.ylabel(f"F{beta} Score"); plt.title(f"F{beta} Score vs Epoch")
            plt.legend(); plt.tight_layout()
            fig3.savefig(f"{out_dir}/F{beta}_Score.pdf"); plt.close(fig3)

            # AUC-PR plot
            fig4 = plt.figure(figsize=(16,9))
            plt.plot(epochs, self.history["train_auc_pr"], label="train AUC-PR")
            plt.plot(epochs, self.history["val_auc_pr"], label="val AUC-PR")
            plt.xlabel("Epoch"); plt.ylabel("AUC-PR"); plt.title("AUC-PR vs Epoch")
            plt.legend(); plt.tight_layout()
            fig4.savefig(f"{out_dir}/AUC_PR.pdf"); plt.close(fig4)

            if self.trues and self.preds:
                y_true  = np.concatenate(self.trues)
                y_score = np.concatenate(self.preds)
                if len(np.unique(y_true)) > 1:
                    prc = precision_recall_curve(y_true, y_score)
                    precision, recall, _ = prc
                    torch.save(prc, f"{trainer.logger.log_dir}/prc.pt")

                    fig5=plt.figure(figsize=(16,9))
                    plt.plot(recall, precision, lw=2)
                    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Validation Precision–Recall Curve")
                    plt.grid(True, linestyle='--',linewidth=0.3,color='gray',alpha=0.7)
                    plt.tight_layout()
                    fig5.savefig(f"{out_dir}/PRC.pdf"); plt.close(fig5)
                else:
                     print("PlotMetrics: PRC plot skipped, only one class present in validation set.")
            else:
                print("PlotMetrics: PRC plot skipped, no prediction data collected.")
            p_at99_r = BinaryPrecisionAtFixedRecall(min_recall= 0.99)
            r_at95_p = BinaryRecallAtFixedPrecision(min_precision = 0.95)
            p_at_r, _ = p_at99_r(torch.tensor(y_score),torch.tensor(y_true))
            r_at_p, _ = r_at95_p(torch.tensor(y_score),torch.tensor(y_true))
            
            try:
                if trainer.logger is not None:
                    trainer.logger.log_metrics({
                        "p_at99_r": float(p_at_r),
                        "r_at95_p": float(r_at_p)
                    }, step=trainer.current_epoch)
            except Exception as err:
                print(f"PlotMetrics: warning - could not log p@r metrics via logger: {err}")
            
            # Append final r@p, p@r to metrics.csv (too expensive to compute every epoch)
            df = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
            df["p_at99_r"] = [0.] * (len(df) - 1) + [float(p_at_r)]
            df["r_at95_p"] = [0.] * (len(df) - 1) + [float(r_at_p)]
            df.to_csv(f"{trainer.logger.log_dir}/metrics.csv",index=False)

            print(f"  Val precision @ 99 recall: {p_at_r:.3f}")
            print(f"  Val recall @ 95 precision: {r_at_p:.3f}")
        
        elif task == "tracking":
            for pt in pl_module.pt_thres:
                 fig_acc = plt.figure(figsize=(16,9))
                 plt.plot(epochs, self.history[f"train_AP@{pt}"], label=f"train AP@{pt}")
                 plt.plot(epochs, self.history[f"val_AP@{pt}"], label=f"val AP@{pt}")
                 plt.xlabel("Epoch"); plt.ylabel("AP@k"); plt.title(f"AP@{pt} vs Epoch")
                 plt.legend(); plt.tight_layout()
                 fig_acc.savefig(f"{out_dir}/AP_{pt}.pdf"); plt.close(fig_acc)

        print(f"Plots saved to {out_dir}")

    def on_test_epoch_end(self, trainer, pl_module):
        if not self.history:
            return

        task = pl_module.hparams.task
        if task != "pileup":
            return
        
        if self.test_trues and self.test_preds:
            y_true  = np.concatenate(self.test_trues)
            y_score = np.concatenate(self.test_preds)

            p_at99_r = BinaryPrecisionAtFixedRecall(min_recall=0.99)
            r_at95_p = BinaryRecallAtFixedPrecision(min_precision=0.95)
            p_val, _ = p_at99_r(torch.tensor(y_score), torch.tensor(y_true))
            r_val, _ = r_at95_p(torch.tensor(y_score), torch.tensor(y_true))

            if trainer.logger is not None:
                trainer.logger.log_metrics({"test_p_at99_r": float(p_val), "test_r_at95_p": float(r_val)}, step=trainer.current_epoch)
            
            print(f"  Test precision @ 99 recall: {p_val:.3f}")
            print(f"  Test recall @ 95 precision: {r_val:.3f}")

class HistLogger(Callback):
    """
    Logs MLP and W_out embeddings to TensorBoard each N epochs.
    """

    def __init__(self, sharded: bool):
        self.sharded = sharded
        self.logged_flag = False
        self._lat_W, self._lat_M, self._labels = None, None, None
        self._hooks = []
        self._MAX_POINTS = 100000
        self._every_N = 10

    @staticmethod
    def _tb_loggers(trainer):
        raw = trainer.loggers if hasattr(trainer, "loggers") else [trainer.logger]
        return [lg for lg in raw if lg.__class__.__name__.startswith("TensorBoard")]
    
    @staticmethod
    def _find_module(pl_module, target_name: str):
        for n, m in pl_module.named_modules():
            if n.endswith(target_name):
                return m
        return None
    
    def on_fit_start(self, trainer, pl_module):
        mod_W  = self._find_module(pl_module, "W")
        mod_M  = self._find_module(pl_module, "mlp_out")
        self._hooks = [
            mod_W.register_forward_hook(
                lambda _, __, out: setattr(self, "_lat_W",
                                           out.detach().cpu()[: self._MAX_POINTS])),
            mod_M.register_forward_hook(
                lambda _, __, out: setattr(self, "_lat_M",
                                           out.detach().cpu()[: self._MAX_POINTS])),
        ]

    def on_fit_end(self, *_):
        for h in self._hooks:
            h.remove()

    def on_train_epoch_start(self, trainer, pl_module):
        self.logged_flag = False
        self._labels = None 

    def on_validation_epoch_start(self, *_):
        self._lat_W, self._lat_M = None, None
    
    def on_validation_batch_start(self, trainer, pl_module, batch, *_):
        if self._labels is None:
            task = pl_module.hparams.task
            if task == "tracking":
                self._labels = batch.particle_id.detach().cpu()
            elif task == "pileup":
                self._labels = batch.y.detach().cpu()
            else: # Unknown task
                self._labels = None

    def on_after_backward(self, trainer, pl_module):
        if self.logged_flag:
            return
        self.logged_flag = True

        tb_loggers = self._tb_loggers(trainer)
        if not tb_loggers:                     # nothing to write to
            return

        gather_ctx = (
            FSDP.summon_full_params(pl_module, recurse=True, writeback=False)
            if self.sharded else nullcontext()
        )

        # gather parameters (no‑op for non‑FSDP models)
        with gather_ctx:
            for name, param in pl_module.named_parameters():
                for tb in tb_loggers:
                    if param.grad is None or param.grad.numel() == 0:
                        continue
                    weights = param.detach()
                    rank_zero_only(tb.experiment.add_histogram)(
                        name, weights, trainer.current_epoch
                    )
                    if param.grad is not None:
                        rank_zero_only(tb.experiment.add_histogram)(
                            f"{name}.grad", param.grad.detach(), trainer.current_epoch
                        )
                        
    def on_validation_epoch_end(self, trainer, pl_module):
        is_final = trainer.current_epoch == (trainer.max_epochs - 1)
        if (trainer.current_epoch % self._every_N) and not is_final:
            return
        
        if pl_module.global_rank != 0:
            return
        if self._lat_W is None or self._lat_M is None:
            return

        # prepare metadata = class labels (as strings)
        # _lat_M is the MLP_out embedding, potentially sliced by _MAX_POINTS
        n_pts_to_log_mlp = self._lat_M.shape[0]
        meta_mlp = None
        final_embedding_mlp = self._lat_M.float()

        if self._labels is not None:
            num_labels_available = self._labels.numel()
            points_in_embedding_mlp = final_embedding_mlp.shape[0]

            if num_labels_available < points_in_embedding_mlp:
                warnings.warn(
                    f"HistLogger: Truncating MLP_out embeddings from {points_in_embedding_mlp} to {num_labels_available} to match available labels."
                )
                final_embedding_mlp = final_embedding_mlp[:num_labels_available]
                meta_mlp = [str(l.item()) for l in self._labels[:num_labels_available]]
            elif num_labels_available > points_in_embedding_mlp:
                # This case should ideally not happen if _MAX_POINTS correctly caps embeddings
                # and labels are from the same data that produced those embeddings.
                warnings.warn(
                    f"HistLogger: Have more labels ({num_labels_available}) than MLP_out embedding points ({points_in_embedding_mlp}). Using labels for available points only."
                )
                meta_mlp = [str(l.item()) for l in self._labels[:points_in_embedding_mlp]]
            else: # num_labels_available == points_in_embedding_mlp
                meta_mlp = [str(l.item()) for l in self._labels[:num_labels_available]] # or [:points_in_embedding_mlp]
        else:
            warnings.warn("HistLogger: self._labels is None. No metadata for MLP_out projector.")

        # For W_out embedding (usually has same number of points as MLP_out if they operate on same items)
        n_pts_to_log_w = self._lat_W.shape[0]
        meta_w = None
        final_embedding_w = self._lat_W.float()

        if self._labels is not None:
            num_labels_available = self._labels.numel()
            points_in_embedding_w = final_embedding_w.shape[0]

            if num_labels_available < points_in_embedding_w:
                warnings.warn(
                    f"HistLogger: Truncating W_out embeddings from {points_in_embedding_w} to {num_labels_available} to match available labels."
                )
                final_embedding_w = final_embedding_w[:num_labels_available]
                meta_w = [str(l.item()) for l in self._labels[:num_labels_available]]
            elif num_labels_available > points_in_embedding_w:
                warnings.warn(
                    f"HistLogger: Have more labels ({num_labels_available}) than W_out embedding points ({points_in_embedding_w}). Using labels for available points only."
                )
                meta_w = [str(l.item()) for l in self._labels[:points_in_embedding_w]]
            else: # num_labels_available == points_in_embedding_w
                meta_w = [str(l.item()) for l in self._labels[:num_labels_available]]
        # If self._labels was None, meta_w will remain None, which is handled by add_embedding.

        writer = self._tb_loggers(trainer)[0].experiment
        writer.add_embedding(final_embedding_w,
                             metadata=meta_w, tag="W_out",
                             global_step=trainer.current_epoch)
        writer.add_embedding(final_embedding_mlp,
                             metadata=meta_mlp, tag="MLP_out",
                             global_step=trainer.current_epoch)
        writer.flush()
        
class SignalExit(L.Callback):
    """
    Gracefully stop training + execute post-processing/cleanup by creating a STOP file.
    Simply "touch STOP" in the root directory to trigger.
    """
    def __init__(self, stop_file_path="STOP"):
        super().__init__()
        self.stop_file_path = stop_file_path
        self.should_stop = False
        self._cleanup_stop_file()
        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            print(f"To gracefully stop training, create file: touch {self.stop_file_path}")
    
    def _cleanup_stop_file(self):
        """Remove stop file if it exists from previous runs"""
        if (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
            if os.path.exists(self.stop_file_path):
                os.remove(self.stop_file_path)
    
    def _check_stop_file_distributed(self):
        should_stop_local = False
        if not dist.is_available() or not dist.is_initialized():
            # Single GPU case
            if os.path.exists(self.stop_file_path):
                should_stop_local = True
                if not self.should_stop:  # Only print once
                    print(f"\nStop file detected ({self.stop_file_path}). Finishing current epoch...")
                try:
                    os.remove(self.stop_file_path)
                except:
                    pass
        else:
            # Multi-GPU case
            if dist.get_rank() == 0:
                if os.path.exists(self.stop_file_path):
                    should_stop_local = True
                    if not self.should_stop:  # Only print once
                        print(f"\nStop file detected ({self.stop_file_path}). Finishing current epoch...")
                    try:
                        os.remove(self.stop_file_path)
                    except:
                        pass
            
            # Broadcast the stop signal from rank 0 to all other ranks
            should_stop_tensor = torch.tensor(int(should_stop_local), device='cuda' if torch.cuda.is_available() else 'cpu')
            dist.broadcast(should_stop_tensor, src=0)
            should_stop_local = bool(should_stop_tensor.item())
        
        if should_stop_local:
            self.should_stop = True
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Only check every N batches to reduce overhead
        if batch_idx % 10 == 0:
            self._check_stop_file_distributed()
        
        if self.should_stop:
            if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
                print("Stopping training gracefully after current epoch...")
            trainer.should_stop = True
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx % 10 == 0:
            self._check_stop_file_distributed()
        
        if self.should_stop:
            trainer.should_stop = True
    
    def on_fit_end(self, trainer, pl_module):
        self._cleanup_stop_file()
    
@rank_zero_only
def _log_model_info(model, dataset, N=120000, k=1):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"Number of trainable parameters: {num_params:,}")
    count_flops_and_params(model, dataset, N, k)

@torch.no_grad()
def count_flops_and_params(model, dataset, N, k):
    E = k * N
    x = torch.randn((N, dataset.x_dim))
    edge_index = torch.randint(0, N, (2, E))
    coords = torch.randn((N, dataset.coords_dim))
    pos = coords[..., :2]
    batch = torch.zeros(N, dtype=torch.long)
    edge_weight = torch.randn((E, 1))

    if dataset.task == "pileup" or dataset.task == "tracking":
        x[..., -2:] = 0.0

    data = {"x": x, "edge_index": edge_index, "coords": coords, "pos": pos, "batch": batch, "edge_weight": edge_weight}
    print(flop_count_table(FlopCountAnalysis(model, data), max_depth=1))