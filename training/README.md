# Novyrix scratch training

Novyrix's text models use a custom decoder-only Transformer initialized from random weights. The training and inference paths do not load any pretrained language-model checkpoint.

## Train the three Novyrix tiers

From the project root, activate the virtual environment and run:

```bash
./train_all.sh
```

The script uses pinned public examples by default, or datasets supplied through
`NOVYRIX_DATASET_4_0`, `NOVYRIX_DATASET_5_0`, and `NOVYRIX_DATASET_6_0`.
For a pipeline-only experiment using only the tiny synthetic examples, opt out
of public data explicitly:

```bash
NOVYRIX_USE_DEMO_DATA=true NOVYRIX_USE_PUBLIC_DATA=false ./train_all.sh
```

Small corpora trigger a warning; they cannot produce broad assistant capability.
Only train on data you have permission to use.

This generates small demonstration datasets and trains separate architectures:

- 4.0: width 256, 4 layers, 4 attention heads
- 5.0: width 384, 6 layers, 6 attention heads
- 6.0: width 512, 8 layers, 8 attention heads

By default, `train_all.sh` prepares pinned public corpora and uses tier-specific
mixtures:

- 4.0: reviewed English OpenAssistant conversations plus Novyrix examples
- 5.0: general conversation data plus syntax-checked Python instruction data
- 6.0: the broadest mixture here—general conversations, Python examples, and
  Novyrix examples

The current filtered corpora contain 9,018 conversation examples and 20,024
Python outputs that pass Python syntax parsing. Syntax validity does not prove
that code is correct. Source, pinned revision, declared license, and filtering
details are recorded in `training/data/prepared/renamed/SOURCES*.json` and copied into
each training manifest. Raw and prepared corpora are git-ignored; their source
README/license files remain under `training/data/external/`.

By default, the script also prepares exact-size web-text pools:

- 4.0: 2,000,000 records from arXiv abstracts whose per-record metadata says CC0.
- 5.0: 2,000,000 of those abstracts plus 2,000,000 PubMed Central records marked Public Domain, CC0, or attribution-only CC BY.
- 6.0: 2,000,000 arXiv abstracts, 3,000,000 similarly license-checked PubMed records, 3,200,000 public-domain Caselaw Access Project records, and 1,800,000 US Government Publishing Office records marked Public Domain.

The Common Pile datasets are filtered source collections; the preparation script
streams them, checks each record's license metadata, records the source revision
and URL, and stops if it cannot reach the requested count. CC BY attribution is
preserved in each training record and source manifest. The script does not scrape
arbitrary websites or duplicate rows to fill a quota. Each web record is capped
at 2,048 UTF-8 bytes and trains as plain-text next-byte prediction. This is
pretraining text, not a set of verified question-and-answer pairs. The trainer
keeps large JSONL corpora on disk and samples from them instead of loading
millions of tensor objects into RAM. When both are present, the default mix
favors supervised chat and instruction examples: 65% for 4.0, 60% for 5.0,
and 55% for 6.0. 4.0 includes syntax-checked Python instruction examples
alongside general conversations. Configure these ratios with
`NOVYRIX_INSTRUCTION_FRACTION_4_0`, `_5_0`, and `_6_0`.
Reviewed corrections for observed tier-specific failures live in
`training/manual_examples/` and are included when public corpora are prepared;
the training script also includes them when public data is disabled.
At runtime, Novyrix can answer a close match from that same tier's reviewed
correction directly. This is a small local fallback for those specific tasks;
other requests still use the selected checkpoint and its unreadable-output guard.

These are corpus sizes, not a promise that each record is seen during one run.
With the default step budgets, the trainer samples only a small fraction of the
millions of available records. Reaching a full pass at the current batch sizes
would require hundreds of thousands to over a million optimizer steps. More
records alone do not guarantee accurate answers.

Disable public data with `NOVYRIX_USE_PUBLIC_DATA=false` and provide your own
licensed corpora. The default training budgets scale by tier: 5,000 steps for
4.0, 12,000 for 5.0, and 24,000 for 6.0, with context lengths of 512/1024/2048
byte tokens. Because the tokenizer is byte-based, these are byte windows, not
word or conventional subword-token counts. Training uses smaller per-step batches
and gradient accumulation for longer windows to control memory. Validation selects the candidate
checkpoint, and the existing checkpoint is not replaced until the run finishes.
If interrupted, the previous checkpoint remains usable and the training
manifest stays marked `in_progress`. Restart the API after training so its
in-memory model cache loads the new checkpoint.

Override all tiers with `NOVYRIX_TRAIN_STEPS`, or a single tier with
`NOVYRIX_TRAIN_STEPS_4_0`, `NOVYRIX_TRAIN_STEPS_5_0`, or
`NOVYRIX_TRAIN_STEPS_6_0`. `NOVYRIX_BATCH_SIZE`, `NOVYRIX_GRAD_ACCUM`,
`NOVYRIX_CONTEXT_LENGTH`, `NOVYRIX_LR`, `NOVYRIX_EVAL_EVERY`,
`NOVYRIX_EVAL_BATCHES`, and `NOVYRIX_EARLY_STOP_PATIENCE` tune the run.
Set patience to `0` to disable early stopping. Training time depends on
hardware; a dry run prints the plan without starting training or changing
checkpoints:

```bash
NOVYRIX_DRY_RUN=true ./train_all.sh
```

The model uses a reversible UTF-8 byte tokenizer. Chat JSONL records with
`messages`, `prompt`/`response`, or serialized `<|im_start|>assistant` turns
train loss on assistant answers rather than teaching the model to reproduce
the user's prompt. Plain text files use next-token training. Validation holds
out whole examples, and the checkpoint is saved when validation loss improves.
Each output folder contains a checkpoint, model configuration, tokenizer
description, and training manifest. The API runtime loads only these custom
scratch checkpoints.

Generate a continuation manually:

```bash
python3 training/generate_scratch.py \
  --checkpoint training/output/novyrix-tier-4.0-scratch/checkpoint.pt \
  --prompt "Hello"
```

## Limitations

These remain small randomly initialized models. 4.0 targets routine tasks and
basic code; 5.0 gets extra coding data and more room for implementation/debugging;
6.0 gets the broadest training mix, longest configured context, and most compute.
The tiers also have distinct response budgets. Longer context settings require
retraining to take effect because existing checkpoints retain their saved
context length. Longer windows increase memory and training time; they do not by
themselves guarantee better reasoning or correct code.
That is a relative specialization, not a promise that 6.0 can answer anything.
Broad capability requires far more high-quality data, compute, and held-out
evaluation. Never rely on a checkpoint based only on falling training loss.

