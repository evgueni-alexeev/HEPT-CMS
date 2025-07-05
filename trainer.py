import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import yaml
import argparse

from pathlib import Path
from typing import Tuple, Optional, Union
import ast
from functools import partial
import torch
import lightning as L

from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning_fabric.utilities.rank_zero import rank_zero_only
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from lightning_modules import TrackLightning as PL
from lightning_modules import TrackDataModule as DM
from lightning_modules import VRAMTracker, PlotMetrics, HistLogger, SignalExit

def ckpt_dir(run_dir: Union[str,Path] , prefer: Optional[str] = None) -> Tuple[Optional[Path], Optional[Path]]:
    """Find the latest checkpoint (override with 'prefer') and hparams file in a directory."""
    run_dir  = Path(run_dir).expanduser().resolve()
    hparams_path = run_dir / "hparams.yaml"
    if not hparams_path.is_file():
         hparams_list = list(run_dir.glob("**/hparams.yaml"))
         hparams_path = hparams_list[0] if hparams_list else None

    ckpts = sorted(run_dir.glob("**/*.ckpt"), key=lambda p: p.stat().st_mtime)
    ckpt_path = None
    if ckpts:
        if prefer:
            # Search for preferred checkpoint name (e.g., 'last.ckpt')
            target = next((p for p in ckpts if p.name == prefer), None)
            ckpt_path = target or ckpts[-1]
        else:
            ckpt_path = ckpts[-1]

    return ckpt_path, hparams_path

def run_lightning(config):
    L.seed_everything(config["seed"])
    assert "task" in config
    task = config["task"]
    print(f"Running task: {task}")

    data_module = DM(config)
    if config["resume"] == False:
        model = PL(config)
        resume_ckpt = None
        
        mc  = ModelCheckpoint(
            monitor=config["main_metric"],
            mode="max",
            filename="{epoch}-{val_loss:.4f}-" + f"{{{config['main_metric']}:.4f}}",
            save_last=True,
            every_n_epochs = 1,
            verbose=True
        )
        mc_loss = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="BEST_VAL_LOSS-{epoch}-{val_loss:.4f}",
            save_last=False,
            every_n_epochs = 1,
            verbose=True
        )

        rank_zero_only(print(f"Starting new run, with {config['num_events']} events, for {config['num_epochs']} epochs."))
    else:
        ckpt_directory = Path(config["resume"])
        assert ckpt_directory.exists(), f"Checkpoint path {ckpt_directory} not found!"
        resume_ckpt, resume_config = ckpt_dir(ckpt_directory)
        config = yaml.safe_load(resume_config.open("r").read())
        model = PL(config)
        ckpt = torch.load(resume_ckpt, map_location="cpu")
        callback_params = list(ckpt["callbacks"].keys())[0]
        mc = ModelCheckpoint(**ast.literal_eval(callback_params[callback_params.find("{"):]))
        mc_loss = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            filename="BEST_VAL_LOSS-{epoch}-{val_loss:.4f}",
            save_last=False,
            every_n_epochs = 1,
            verbose=True
        )
        epoch_saved = ckpt.get("epoch", None)

    policy = size_based_auto_wrap_policy
    strat = FSDPStrategy(
        auto_wrap_policy=partial(policy,min_num_params=10000),
        activation_checkpointing_policy=partial(policy,min_num_params=10000),
        sharding_strategy="FULL_SHARD",
        cpu_offload=True,
        use_orig_params=True,
    )
    csv_logger = CSVLogger(
        save_dir="logs/tracking" if task=="tracking" else "logs/pileup", name="csv"
    )
    tb_logger = TensorBoardLogger(
        save_dir = "logs/tracking" if task=="tracking" else "logs/pileup", name="tb", log_graph=False, default_hp_metric=False
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch",log_momentum=False, log_weight_decay=False)
    logger_list = [csv_logger, tb_logger]
    if config["log_tb"]:
        cbs = [mc, mc_loss, lr_monitor, VRAMTracker(), PlotMetrics(), HistLogger(sharded = True), SignalExit()] if config["multiple_gpus"] else [mc, mc_loss, lr_monitor, PlotMetrics(), HistLogger(sharded = False), SignalExit()]
    else:
        cbs = [mc, mc_loss, lr_monitor, VRAMTracker(), PlotMetrics(), SignalExit()] if config["multiple_gpus"] else [mc, mc_loss, lr_monitor, PlotMetrics(), SignalExit()]
    trainer = L.Trainer(
        accelerator=config["device"],
        devices=config["cudacores"] if config["multiple_gpus"] else 1,
        strategy=strat if config["multiple_gpus"] else "auto", 
        max_epochs=config["num_epochs"], #epoch_saved + X
        callbacks = cbs,
        enable_progress_bar=True,
        log_every_n_steps=1,
        precision="32",
        num_sanity_val_steps=0,
        logger=logger_list
    )
    trainer.fit(model, data_module, ckpt_path = resume_ckpt)

    _print_summary(trainer.callback_metrics.items(),config)

    # ----------------------------------------------------

    TEST_RUN = True
    if TEST_RUN:
        # Run testing in a separate Trainer instance with logging disabled
        test_trainer = L.Trainer(
            accelerator=config["device"],
            devices=config["cudacores"] if config["multiple_gpus"] else 1,
            strategy=strat if config["multiple_gpus"] else "auto",
            logger=False,
            enable_progress_bar=True,
            precision="32",
            num_sanity_val_steps=0,
        )
        test_trainer.test(model, data_module, ckpt_path=resume_ckpt, verbose=True)

@rank_zero_only
def _print_summary(its, conf):
    print(conf["model_kwargs"]) 
    print("Last-epoch metrics:")
    for name, val in its:
        print(f"  {name}: {val:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Run pileup or tracking training")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t", "--tracking", action="store_true", help="Use tracking config (config_tracking.yaml)")
    group.add_argument("-p", "--pileup", action="store_true", help="Use pileup config (config_pileup.yaml) [default]")
    args = parser.parse_args()

    # Select config file based on flag
    if args.tracking:
        cfg_file = "cfg_tracking.yaml"
    elif args.pileup:
        cfg_file = "cfg_pileup.yaml"
    else:
        raise ValueError("Must specify either tracking (-t) or pileup (-p)")

    config_path = Path(__file__).parent / cfg_file
    assert config_path.exists(), f"Config file {config_path} not found."
    config = yaml.safe_load(config_path.open("r").read())

    # Prepare cosine LR schedule parameters (global steps)
    if config.get("lr_scheduler_name") == "cosine":
        steps_per_epoch = int(0.8 * config["num_events"] / config["batch_size"])
        total_steps = steps_per_epoch * config["num_epochs"]
        lr_kw = config.setdefault("lr_scheduler_kwargs", {})
        warmup_steps = lr_kw.get("warmup_epochs", max(1, int(0.05 * config["num_epochs"]))) * steps_per_epoch
        lr_kw["num_training_steps"] = total_steps
        lr_kw["num_warmup_steps"] = warmup_steps
        lr_kw["eta_min"] = lr_kw.get("min_lr", 0.0)
        lr_kw.pop("warmup_epochs", None)
        lr_kw.pop("min_lr", None)

    run_lightning(config)

if __name__ == "__main__":
    main()