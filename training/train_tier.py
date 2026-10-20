"""Compatibility entry point for training one Novyrix tier from scratch."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scratch_lm import main as scratch_main


def main() -> None:
    tier = os.environ.get("NOVYRIX_TIER", "3.2")
    dataset = Path(os.environ.get("NOVYRIX_DATASET", ""))
    output = Path(os.environ.get("NOVYRIX_OUTPUT", ""))
    if not dataset.is_file():
        raise FileNotFoundError(f"Training dataset not found: {dataset}")
    if not output:
        raise ValueError("NOVYRIX_OUTPUT must name a scratch checkpoint directory")

    sys.argv = [
        "scratch_lm.py",
        "--corpus", str(dataset),
        "--output-dir", str(output),
        "--steps", os.environ.get("NOVYRIX_TRAIN_STEPS", "1000"),
        "--eval-every", os.environ.get("NOVYRIX_EVAL_EVERY", "100"),
        "--batch-size", os.environ.get("NOVYRIX_BATCH_SIZE", "8"),
        "--context-length", os.environ.get("NOVYRIX_CONTEXT_LENGTH", "256"),
        "--width", os.environ.get("NOVYRIX_MODEL_WIDTH", "256"),
        "--layers", os.environ.get("NOVYRIX_MODEL_LAYERS", "4"),
        "--heads", os.environ.get("NOVYRIX_MODEL_HEADS", "4"),
        "--learning-rate", os.environ.get("NOVYRIX_LR", "3e-4"),
        "--seed", os.environ.get("NOVYRIX_SEED", "42"),
    ]
    print(f"Training Novyrix {tier} from random initialization; no pretrained model is used.")
    scratch_main()


if __name__ == "__main__":
    main()
