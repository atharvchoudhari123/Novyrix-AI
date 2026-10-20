#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python virtual environment not found at $PYTHON"
  echo "Create it with: python3 -m venv .venv && source .venv/bin/activate"
  exit 1
fi

export PYTHONUNBUFFERED=1
DATA_DIR="$ROOT/training/data"
OUTPUT_DIR="$ROOT/training/output"
mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

echo "Preparing Novyrix tier-specific training examples."
"$PYTHON" "$ROOT/training/generate_tier_data.py"

for tier in 3.2 4.0 5.7; do
  case "$tier" in
    3.2) width=256; layers=4; heads=4; count=300 ;;
    4.0) width=384; layers=6; heads=6; count=450 ;;
    5.7) width=512; layers=8; heads=8; count=600 ;;
  esac

  safe_tier="${tier//./_}"
  dataset="$DATA_DIR/novyrix_${safe_tier}.jsonl"
  output="$OUTPUT_DIR/novyrix-$tier-scratch"

  echo
  echo "========================================"
  echo "Training Novyrix $tier from random initialization"
  echo "No pretrained language-model weights are used."
  echo "Dataset: $dataset ($count examples)"
  echo "Architecture: width=$width layers=$layers heads=$heads"
  echo "Output: $output"
  echo "========================================"

  NOVYRIX_TIER="$tier" \
  NOVYRIX_DATASET="$dataset" \
  NOVYRIX_OUTPUT="$output" \
  NOVYRIX_MODEL_WIDTH="$width" \
  NOVYRIX_MODEL_LAYERS="$layers" \
  NOVYRIX_MODEL_HEADS="$heads" \
  "$PYTHON" "$ROOT/training/train_tier.py"
done

if [[ -f "$ROOT/.env" ]]; then
  "$PYTHON" - <<'PY'
from pathlib import Path
import re

env = Path(".env")
text = env.read_text(encoding="utf-8")
updates = {
    "NOVYRIX_3_2_CHECKPOINT": "./training/output/novyrix-3.2-scratch",
    "NOVYRIX_4_0_CHECKPOINT": "./training/output/novyrix-4.0-scratch",
    "NOVYRIX_5_7_CHECKPOINT": "./training/output/novyrix-5.7-scratch",
}
for key, value in updates.items():
    pattern = rf"(?m)^{re.escape(key)}=.*$"
    line = f"{key}={value}"
    if re.search(pattern, text):
        text = re.sub(pattern, line, text)
    else:
        text += ("\n" if text and not text.endswith("\n") else "") + line + "\n"
env.write_text(text, encoding="utf-8")
print("Updated local .env with Novyrix scratch checkpoint paths.")
PY
fi

echo
echo "========================================"
echo " ALL NOVYRIX SCRATCH TIERS TRAINED"
echo "========================================"
echo "3.2 -> training/output/novyrix-3.2-scratch (300 examples)"
echo "4.0 -> training/output/novyrix-4.0-scratch (450 examples)"
echo "5.7 -> training/output/novyrix-5.7-scratch (600 examples)"
echo ""
echo "Initial capabilities depend on the small training corpus and compute; evaluate before relying on the models."

