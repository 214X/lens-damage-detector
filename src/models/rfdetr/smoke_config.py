from src.models.rfdetr.config import RFDETR_TRAIN_ARGS


# macOS / MPS-friendly smoke overrides.
# Purpose: validate the full training pipeline with minimal memory pressure.
RFDETR_SMOKE_TRAIN_ARGS = RFDETR_TRAIN_ARGS.copy()
RFDETR_SMOKE_TRAIN_ARGS.update(
    {
        "epochs": 3,
        "batch_size": 2,
        "grad_accum_steps": 1,
        "resolution": 640,
        "num_workers": 0,
        "pin_memory": False,
        "persistent_workers": False,
        "prefetch_factor": None,
        "use_ema": False,
        "gradient_checkpointing": True,
        "eval_max_dets": 100,
        "num_select": 100,
        "log_per_class_metrics": False,
        "compute_test_loss": False,
    }
)
