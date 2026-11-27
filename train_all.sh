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
DATA_DIR="$ROOT/training/data/renamed"
OUTPUT_DIR="$ROOT/training/output"
PUBLIC_DATA_DIR="$ROOT/training/data/prepared/renamed"
WEB_DATA_DIR="$PUBLIC_DATA_DIR/web"
DRY_RUN="${NOVYRIX_DRY_RUN:-false}"
USE_PUBLIC_DATA="${NOVYRIX_USE_PUBLIC_DATA:-true}"
if [[ "$DRY_RUN" != "true" ]]; then
  mkdir -p "$DATA_DIR" "$OUTPUT_DIR"
fi

if [[ "${NOVYRIX_USE_DEMO_DATA:-false}" == "true" ]]; then
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN: would generate small synthetic demonstration datasets."
  else
    echo "Generating small synthetic demonstration datasets (not capability training data)."
    "$PYTHON" "$ROOT/training/generate_tier_data.py"
  fi
else
  echo "Demo-data generation is off; using existing or explicitly configured corpora."
fi

if [[ "$USE_PUBLIC_DATA" == "true" ]]; then
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN: public corpora would be prepared in $PUBLIC_DATA_DIR."
  else
    echo "Preparing pinned Apache-2.0 conversation and MIT Python instruction examples."
    "$PYTHON" "$ROOT/training/prepare_public_data.py"
  fi
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "DRY RUN: license-checked web corpora would be streamed: 4.0=2,000,000; 5.0=4,000,000; 6.0=10,000,000 examples."
  else
    echo "Preparing exact-size web corpora from record-level CC0/public-domain sources."
    "$PYTHON" "$ROOT/training/prepare_web_corpora.py"
  fi
fi

for tier in 4.0 5.0 6.0; do
  safe_tier="${tier//./_}"
  case "$tier" in
    4.0)
      width=256; layers=4; heads=4; context=512; steps="${NOVYRIX_TRAIN_STEPS_4_0:-${NOVYRIX_TRAIN_STEPS:-5000}}"; batch=4; accum=2; lr=3e-4; instruction_fraction="${NOVYRIX_INSTRUCTION_FRACTION_4_0:-0.65}"
      web_shards=("$WEB_DATA_DIR/arxiv_base.jsonl")
      source_manifest_paths=()
      if [[ -n "${NOVYRIX_DATASET_4_0:-}" ]]; then
        IFS=':' read -r -a dataset_paths <<< "$NOVYRIX_DATASET_4_0"
      elif [[ "$USE_PUBLIC_DATA" == "true" ]]; then
        dataset_paths=("$PUBLIC_DATA_DIR/novyrix_4_0.jsonl")
      else
        dataset_paths=("$DATA_DIR/novyrix_4_0.jsonl")
      fi
      if [[ "$USE_PUBLIC_DATA" == "true" ]]; then
        dataset_paths+=("${web_shards[@]}")
        source_manifest_paths+=("$PUBLIC_DATA_DIR/SOURCES_4_0.json" "$WEB_DATA_DIR/SOURCES_4_0.json")
      else
        dataset_paths+=("$ROOT/training/manual_examples/novyrix_4_0.jsonl")
      fi ;;
    5.0)
      width=384; layers=6; heads=6; context=1024; steps="${NOVYRIX_TRAIN_STEPS_5_0:-${NOVYRIX_TRAIN_STEPS:-12000}}"; batch=2; accum=4; lr=2.5e-4; instruction_fraction="${NOVYRIX_INSTRUCTION_FRACTION_5_0:-0.60}"
      web_shards=("$WEB_DATA_DIR/arxiv_base.jsonl" "$WEB_DATA_DIR/pubmed_base.jsonl")
      source_manifest_paths=()
      if [[ -n "${NOVYRIX_DATASET_5_0:-}" ]]; then
        IFS=':' read -r -a dataset_paths <<< "$NOVYRIX_DATASET_5_0"
      elif [[ "$USE_PUBLIC_DATA" == "true" ]]; then
        dataset_paths=("$PUBLIC_DATA_DIR/novyrix_5_0.jsonl")
      else
        dataset_paths=("$DATA_DIR/novyrix_5_0.jsonl")
      fi
      if [[ "$USE_PUBLIC_DATA" == "true" ]]; then
        dataset_paths+=("${web_shards[@]}")
        source_manifest_paths+=("$PUBLIC_DATA_DIR/SOURCES_5_0.json" "$WEB_DATA_DIR/SOURCES_5_0.json")
      else
        dataset_paths+=("$ROOT/training/manual_examples/novyrix_5_0.jsonl")
      fi ;;
    6.0)
      width=512; layers=8; heads=8; context=2048; steps="${NOVYRIX_TRAIN_STEPS_6_0:-${NOVYRIX_TRAIN_STEPS:-24000}}"; batch=1; accum=8; lr=2e-4; instruction_fraction="${NOVYRIX_INSTRUCTION_FRACTION_6_0:-0.55}"
      web_shards=("$WEB_DATA_DIR/arxiv_base.jsonl" "$WEB_DATA_DIR/pubmed_base.jsonl" "$WEB_DATA_DIR/pubmed_extra.jsonl" "$WEB_DATA_DIR/caselaw_base.jsonl" "$WEB_DATA_DIR/caselaw_extra.jsonl" "$WEB_DATA_DIR/usgpo.jsonl")
      source_manifest_paths=()
      if [[ -n "${NOVYRIX_DATASET_6_0:-}" ]]; then
        IFS=':' read -r -a dataset_paths <<< "$NOVYRIX_DATASET_6_0"
      elif [[ "$USE_PUBLIC_DATA" == "true" ]]; then
        dataset_paths=("$PUBLIC_DATA_DIR/novyrix_6_0.jsonl")
      else
        dataset_paths=("$DATA_DIR/novyrix_6_0.jsonl")
      fi
      if [[ "$USE_PUBLIC_DATA" == "true" ]]; then
        dataset_paths+=("${web_shards[@]}")
        source_manifest_paths+=("$PUBLIC_DATA_DIR/SOURCES_6_0.json" "$WEB_DATA_DIR/SOURCES_6_0.json")
      else
        dataset_paths+=("$ROOT/training/manual_examples/novyrix_6_0.jsonl")
      fi ;;
  esac

  # Research exports are opt-in and must be explicitly approved. Candidate
  # exports are never consumed by a training run.
  if [[ "${NOVYRIX_INCLUDE_RESEARCH_CACHE:-false}" == "true" ]]; then
    research_examples="$ROOT/training/data/research_cache/novyrix_${safe_tier}_approved.jsonl"
    research_manifest="$ROOT/training/data/research_cache/novyrix_${safe_tier}_approved.manifest.json"
    if [[ ! -f "$research_examples" || ! -f "$research_manifest" ]]; then
      echo "ERROR: approved research-cache data for Novyrix $tier is missing."
      echo "Create candidates, review them, then approve: python3 training/prepare_research_cache.py --tier $tier; edit the candidate JSONL; rerun with --approved --license <accurate-license>."
      exit 1
    fi
    dataset_paths+=("$research_examples")
    source_manifest_paths+=("$research_manifest")
  fi

  output="$OUTPUT_DIR/novyrix-tier-$tier-scratch"
  dataset_lines=0
  missing_dataset=""
  for dataset_path in "${dataset_paths[@]}"; do
    if [[ -f "$dataset_path" ]]; then
      lines="$(wc -l < "$dataset_path" | tr -d ' ')"
      dataset_lines=$((dataset_lines + lines))
    elif [[ "$DRY_RUN" != "true" ]]; then
      missing_dataset="$dataset_path"
      break
    fi
  done
  if [[ -n "$missing_dataset" ]]; then
    echo "ERROR: dataset for Novyrix $tier was not found: $missing_dataset"
    echo "Set NOVYRIX_DATASET_${safe_tier} to a licensed training corpus."
    echo "For a pipeline-only demo, rerun with NOVYRIX_USE_DEMO_DATA=true."
    exit 1
  fi
  if [[ "$DRY_RUN" == "true" ]]; then
    example_lines="planned web corpus target plus curated examples"
  else
    example_lines="$dataset_lines"
  fi
  if [[ "$dataset_lines" -lt 10000 && "$DRY_RUN" != "true" ]]; then
    echo "WARNING: selected corpus has only $dataset_lines lines; this is far too small for broad assistant capability."
  fi

  PATH_SEPARATOR=':'
  dataset="$(IFS="$PATH_SEPARATOR"; printf '%s' "${dataset_paths[*]}")"
  source_manifest="$(IFS="$PATH_SEPARATOR"; printf '%s' "${source_manifest_paths[*]}")"

  echo
  echo "========================================"
  echo "Training Novyrix $tier from random initialization"
  echo "No pretrained language-model weights are used."
  echo "Dataset files: ${#dataset_paths[@]} ($example_lines lines; actual usable examples are validated by the trainer)"
  echo "Architecture: width=$width layers=$layers heads=$heads context=$context"
  echo "Schedule: steps=$steps batch=$batch gradient_accumulation=$accum learning_rate=$lr instruction_fraction=$instruction_fraction early_stop_patience=${NOVYRIX_EARLY_STOP_PATIENCE:-12}"
  echo "Output: $output"
  echo "========================================"

  if [[ "$DRY_RUN" == "true" ]]; then
    continue
  fi

  NOVYRIX_TIER="$tier" \
  NOVYRIX_DATASET="$dataset" \
  NOVYRIX_OUTPUT="$output" \
  NOVYRIX_MODEL_WIDTH="$width" \
  NOVYRIX_MODEL_LAYERS="$layers" \
  NOVYRIX_MODEL_HEADS="$heads" \
  NOVYRIX_TRAIN_STEPS="$steps" \
  NOVYRIX_INSTRUCTION_FRACTION="$instruction_fraction" \
  NOVYRIX_BATCH_SIZE="${NOVYRIX_BATCH_SIZE:-$batch}" \
  NOVYRIX_GRAD_ACCUM="${NOVYRIX_GRAD_ACCUM:-$accum}" \
  NOVYRIX_CONTEXT_LENGTH="${NOVYRIX_CONTEXT_LENGTH:-$context}" \
  NOVYRIX_LR="${NOVYRIX_LR:-$lr}" \
  NOVYRIX_DATA_SOURCES="$source_manifest" \
  "$PYTHON" "$ROOT/training/train_tier.py"
done

if [[ "$DRY_RUN" == "true" ]]; then
  echo "DRY RUN complete; no datasets, checkpoints, or .env files were changed."
  exit 0
fi

if [[ -f "$ROOT/.env" ]]; then
  "$PYTHON" - <<'PY'
from pathlib import Path
import re

env = Path(".env")
text = env.read_text(encoding="utf-8")
updates = {
    "NOVYRIX_MODEL_4_0_CHECKPOINT": "./training/output/novyrix-tier-4.0-scratch",
    "NOVYRIX_MODEL_5_0_CHECKPOINT": "./training/output/novyrix-tier-5.0-scratch",
    "NOVYRIX_MODEL_6_0_CHECKPOINT": "./training/output/novyrix-tier-6.0-scratch",
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
echo "4.0 -> training/output/novyrix-tier-4.0-scratch"
echo "5.0 -> training/output/novyrix-tier-5.0-scratch"
echo "6.0 -> training/output/novyrix-tier-6.0-scratch"
echo ""
echo "Initial capabilities depend on the small training corpus and compute; evaluate before relying on the models."

