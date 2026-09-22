"""
train.py — TFT Benchmark: Model Training
==========================================
Defines the TFT model, trainer, and callbacks.
Saves the best checkpoint to checkpoints/best_model.ckpt

Hyperparameter philosophy:
  - Scale is intentionally modest (hidden_size=32) to reflect
    TFT's role as a benchmark, not an independently tuned model.
  - Justification: see thesis_justification.md §b
"""

import logging
import sys
from pathlib import Path

import lightning.pytorch as pl  # <--- Thay đổi ở đây
import torch
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch.callbacks import (  # <--- Thay đổi ở đây
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger  # <--- Thay đổi ở đây

# Try to import from data_prep module (script mode); fall back to notebook globals (notebook mode)
try:
    from data_prep import prepare_data, set_all_seeds, SEED
except Exception:
    prepare_data = globals().get("prepare_data")
    set_all_seeds = globals().get("set_all_seeds")
    SEED = globals().get("SEED")
    if prepare_data is None or set_all_seeds is None or SEED is None:
        raise ImportError("Could not find prepare_data, set_all_seeds, or SEED in module or notebook globals.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 0.  REPRODUCIBILITY
# ─────────────────────────────────────────────
set_all_seeds(SEED)

# ─────────────────────────────────────────────
# 1.  LOG ENVIRONMENT VERSIONS
# ─────────────────────────────────────────────
def log_versions() -> None:
    import pytorch_forecasting
    logger.info(f"Python              : {sys.version}")
    logger.info(f"PyTorch             : {torch.__version__}")
    logger.info(f"PyTorch Lightning   : {pl.__version__}")
    logger.info(f"PyTorch Forecasting : {pytorch_forecasting.__version__}")
    logger.info(f"CUDA available      : {torch.cuda.is_available()}")

log_versions()

# ─────────────────────────────────────────────
# 2.  PATHS
# ─────────────────────────────────────────────
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = Path("tb_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# 3.  TRAINING HYPERPARAMETERS
# ─────────────────────────────────────────────
# These are fixed — intentional for benchmark fairness.
# Do NOT tune beyond this for the thesis comparison.
TRAIN_CFG = {
    "learning_rate":          1e-3,
    "hidden_size":            16,    
    "attention_head_size":    2,    
    "dropout":                0.15,
    "hidden_continuous_size": 8,     
    "max_epochs":             50,
    "early_stop_patience":    8,
    "reduce_lr_patience":     4,
    "batch_size":             32,    
    "gradient_clip_val":      0.1,
}

# ─────────────────────────────────────────────
# 4.  MODEL DEFINITION
# ─────────────────────────────────────────────

def build_tft(training_dataset) -> TemporalFusionTransformer:
    """
    Instantiate TFT from the training dataset.
    Using .from_dataset() ensures embedding sizes, feature
    encoders, and normalisation are derived from data automatically.
    """
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,

        # Optimisation
        learning_rate=TRAIN_CFG["learning_rate"],
        reduce_on_plateau_patience=TRAIN_CFG["reduce_lr_patience"],

        # Architecture
        hidden_size=TRAIN_CFG["hidden_size"],
        attention_head_size=TRAIN_CFG["attention_head_size"],
        dropout=TRAIN_CFG["dropout"],
        hidden_continuous_size=TRAIN_CFG["hidden_continuous_size"],

        # Loss: QuantileLoss covers median (q=0.5) + PI quantiles.
        # Point forecast for evaluation = q=0.5 output.
        loss=QuantileLoss(quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]),

        # Logging
        log_interval=10,
        log_val_interval=1,
    )

    n_params = sum(p.numel() for p in tft.parameters() if p.requires_grad)
    logger.info(f"TFT parameter count: {n_params:,}")
    logger.info(f"TFT hparams: {tft.hparams}")
    return tft


# ─────────────────────────────────────────────
# 5.  CALLBACKS
# ─────────────────────────────────────────────

def build_callbacks(checkpoint_dir: Path):
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=TRAIN_CFG["early_stop_patience"],
        mode="min",
        verbose=True,
    )

    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="tft_best_{epoch:02d}_{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    return [early_stop, checkpoint, lr_monitor]


# ─────────────────────────────────────────────
# 6.  TRAINER
# ─────────────────────────────────────────────

def build_trainer(checkpoint_dir: Path) -> pl.Trainer:
    tb_logger = TensorBoardLogger(
        save_dir=str(LOG_DIR),
        name="tft_benchmark",
    )

    callbacks = build_callbacks(checkpoint_dir)

    trainer = pl.Trainer(
        max_epochs=TRAIN_CFG["max_epochs"],
        accelerator="auto",           # GPU if available, else CPU
        devices="auto",
        gradient_clip_val=TRAIN_CFG["gradient_clip_val"],
        callbacks=callbacks,
        logger=tb_logger,
        precision="16-mixed",
        enable_model_summary=True,
        log_every_n_steps=10,
        # Deterministic mode for reproducibility
        # (may be slower on GPU but ensures identical results)
        deterministic=True,
    )
    return trainer


# ─────────────────────────────────────────────
# 7.  MAIN TRAINING LOOP
# ─────────────────────────────────────────────

def train() -> dict:
    """
    Full training pipeline. Returns paths and objects needed
    by evaluate.py and interpret.py.
    """
    # Prepare data
    logger.info("=== Preparing data ===")
    data = prepare_data()

    train_loader = data["train_loader"]
    val_loader   = data["val_loader"]
    training_dataset = data["train_dataset"]

    # Build model
    logger.info("=== Building TFT model ===")
    tft = build_tft(training_dataset)

    # Build trainer
    trainer = build_trainer(CHECKPOINT_DIR)

    # Train
    logger.info("=== Starting training ===")
    trainer.fit(
        model=tft,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Locate best checkpoint
    best_ckpt = trainer.checkpoint_callback.best_model_path
    best_val_loss = trainer.checkpoint_callback.best_model_score
    logger.info(f"Best checkpoint : {best_ckpt}")
    logger.info(f"Best val loss   : {best_val_loss:.6f}")

    # Save a symlink / copy for easy access
    best_path = CHECKPOINT_DIR / "best_model.ckpt"
    if not best_path.exists() or str(best_path) != best_ckpt:
        import shutil
        shutil.copy(best_ckpt, best_path)
        logger.info(f"Saved best model → {best_path}")

    return {
        "tft":              tft,
        "trainer":          trainer,
        "best_ckpt":        str(best_path),
        "best_val_loss":    float(best_val_loss),
        "training_dataset": training_dataset,
        "data":             data,
    }


if __name__ == "__main__":
    # Windows yêu cầu guard này cho multiprocessing
    import torch.multiprocessing as mp
    mp.freeze_support()
    
    result = train()
    print(f"\n✓ Training complete. Best checkpoint: {result['best_ckpt']}")