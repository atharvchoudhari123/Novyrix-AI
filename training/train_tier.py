"""Compatibility entry point for training one Novyrix tier from scratch."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scratch_lm import main as scratch_main


def main() -> None:
    tier = os.environ.get("NOVYRIX_TIER", "4.0")
    dataset_paths = [
        Path(item).expanduser()
        for item in os.environ.get("NOVYRIX_DATASET", "").split(os.pathsep)
        if item.strip()
    ]
    output = Path(os.environ.get("NOVYRIX_OUTPUT", ""))
    if not dataset_paths:
        raise FileNotFoundError("NOVYRIX_DATASET must name at least one JSONL or text corpus")
    for dataset in dataset_paths:
        if not dataset.is_file():
            raise FileNotFoundError(f"Training dataset not found: {dataset}")
    if not output:
        raise ValueError("NOVYRIX_OUTPUT must name a scratch checkpoint directory")

    sys.argv = [
        "scratch_lm.py",
        "--corpus", *(str(dataset) for dataset in dataset_paths),
        "--output-dir", str(output),
        *(["--source-manifest", *[
            str(Path(item).expanduser())
            for item in os.environ["NOVYRIX_DATA_SOURCES"].split(os.pathsep)
            if item.strip()
        ]] if os.environ.get("NOVYRIX_DATA_SOURCES") else []),
        "--steps", os.environ.get("NOVYRIX_TRAIN_STEPS", "1000"),
        "--eval-every", os.environ.get("NOVYRIX_EVAL_EVERY", "100"),
        "--eval-batches", os.environ.get("NOVYRIX_EVAL_BATCHES", "20"),
        "--batch-size", os.environ.get("NOVYRIX_BATCH_SIZE", "8"),
        "--context-length", os.environ.get("NOVYRIX_CONTEXT_LENGTH", "256"),
        "--width", os.environ.get("NOVYRIX_MODEL_WIDTH", "256"),
        "--layers", os.environ.get("NOVYRIX_MODEL_LAYERS", "4"),
        "--heads", os.environ.get("NOVYRIX_MODEL_HEADS", "4"),
        "--learning-rate", os.environ.get("NOVYRIX_LR", "3e-4"),
        "--weight-decay", os.environ.get("NOVYRIX_WEIGHT_DECAY", "0.1"),
        "--warmup-steps", os.environ.get("NOVYRIX_WARMUP_STEPS", "100"),
        "--min-lr-ratio", os.environ.get("NOVYRIX_MIN_LR_RATIO", "0.1"),
        "--validation-fraction", os.environ.get("NOVYRIX_VALIDATION_FRACTION", "0.15"),
        "--gradient-accumulation-steps", os.environ.get("NOVYRIX_GRAD_ACCUM", "1"),
        "--early-stopping-patience", os.environ.get("NOVYRIX_EARLY_STOP_PATIENCE", "12"),
        "--min-delta", os.environ.get("NOVYRIX_MIN_DELTA", "0.001"),
        "--seed", os.environ.get("NOVYRIX_SEED", "42"),
    ]
    print(f"Training Novyrix {tier} from random initialization; no pretrained model is used.")
    scratch_main()


if __name__ == "__main__":
    main()
