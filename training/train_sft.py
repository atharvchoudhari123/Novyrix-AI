"""Legacy command retained as a pointer to the Novyrix scratch trainer."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_tier import main as train_tier_main


def main() -> None:
    os.environ.setdefault("NOVYRIX_TIER", "4.0")
    os.environ.setdefault("NOVYRIX_DATASET", str(ROOT / "training/data/renamed/novyrix_4_0.jsonl"))
    os.environ.setdefault("NOVYRIX_OUTPUT", str(ROOT / "training/output/novyrix-tier-4.0-scratch"))
    train_tier_main()


if __name__ == "__main__":
    main()
