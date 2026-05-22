from pathlib import Path

from src.models.rfdetr.config import RFDETR_TRAIN_ARGS
from src.models.rfdetr.model_registry import MODEL_SEQUENCE
from src.models.rfdetr.reporting import (
    collect_environment,
    create_run_dir,
    summarize_fold,
    summarize_run,
    write_json,
)

# ROOT DIRECTORY
ROOT = Path(__file__).resolve().parents[3]
# DATA DIRECTORY
DATA_DIR = ROOT / "data"
# OUTPUT DIRECTORY
RUNS_DIR = ROOT / "runs" / "rfdetr"


def train_fold(fold_dir: Path, output_dir: Path, model_cls, model_label: str):
    """
    Train RF-DETR on a single fold.
    """
    model = model_cls()

    train_args = RFDETR_TRAIN_ARGS.copy()
    train_args["dataset_dir"] = str(fold_dir)
    train_args["output_dir"] = str(output_dir)
    train_args["run"] = f"{model_label}_{fold_dir.name}"

    model.train(**train_args)


def train_5_fold():
    """
    Train RF-DETR on all 5 folds. 
    """
    run_dir = create_run_dir(RUNS_DIR, "full")
    write_json(
        run_dir / "run_config.json",
        {
            "run_type": "full",
            "model_sequence": [label for label, _ in MODEL_SEQUENCE],
            "train_args": RFDETR_TRAIN_ARGS,
        },
    )
    write_json(run_dir / "environment.json", collect_environment("full", DATA_DIR))
    folds = sorted(DATA_DIR.glob("fold_*"))

    for model_label, model_cls in MODEL_SEQUENCE:
        model_run_dir = run_dir / model_label
        model_run_dir.mkdir(parents=True, exist_ok=True)

        for fold_dir in folds:
            fold_name = fold_dir.name
            output_dir = model_run_dir / fold_name

            print(f"\n========== {model_label} | {fold_name} ==========")

            train_fold(
                fold_dir=fold_dir,
                output_dir=output_dir,
                model_cls=model_cls,
                model_label=model_label,
            )
            summarize_fold(output_dir)

        summarize_run(model_run_dir, f"full-{model_label}")



if __name__ == "__main__":
    train_5_fold()
