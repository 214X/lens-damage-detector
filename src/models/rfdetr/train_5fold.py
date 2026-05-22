from pathlib import Path
from rfdetr import RFDETRSmall

from src.models.rfdetr.config import RFDETR_TRAIN_ARGS

# ROOT DIRECTORY
ROOT = Path(__file__).resolve().parents[3]
# DATA DIRECTORY
DATA_DIR = ROOT / "data"
# OUTPUT DIRECTORY
OUTPUT_DIR = ROOT / "runs" / "rfdetr"


def train_fold(fold_dir: Path, output_dir: Path):
    """
    Train RF-DETR on a single fold.
    """
    # determine the model to train
    model = RFDETRSmall()

    # CONFIGURATIONS and HYPERPARAMETERS for training
    train_args = RFDETR_TRAIN_ARGS.copy()
    train_args["dataset_dir"] = str(fold_dir)
    train_args["output_dir"] = str(output_dir)
    train_args["run"] = fold_dir.name

    model.train(**train_args)

def train_5_fold():
    """
    Train RF-DETR on all 5 folds. 
    """

    folds = sorted(DATA_DIR.glob("fold_*"))

    for fold_dir in folds:
        fold_name = fold_dir.name
        output_dir = OUTPUT_DIR / fold_name

        print(f"\n========== {fold_name} ==========")

        train_fold(
            fold_dir=fold_dir,
            output_dir=output_dir
        )



if __name__ == "__main__":
    train_5_fold()
