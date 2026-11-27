# Novyrix
Novyrix is a self-hosted AI platform.

It contains:

- Novyrix API
- Novyrix Engine
- Model registry
- Model runtime
- Training pipeline
- File handling
- Web interface
- Coding Playground
- Docker configuration

## Novyrix models

Novyrix 4.0
Novyrix 5.0
Novyrix 6.0

These tiers use separate Novyrix decoder-only Transformers initialized from
random weights and trained by Novyrix's scratch-training code. The text-model
runtime loads only Novyrix's own scratch checkpoints.
Each tier has a different Novyrix architecture size; this is ownership of the
model implementation and weights, not a claim of frontier-level capability.
The image and video generators remain separate third-party pretrained models.

## Architecture

```text
Browser
    |
    v
Novyrix API
    |
    v
Novyrix Engine
    |
    +-- Memory
    |
    +-- Context
    |
    +-- Files
    |
    +-- Model Router
    |
    v
Novyrix Runtime
    |
    v
Novyrix Model
```

## Install

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train all Novyrix model tiers

Run the unified training script with existing licensed corpora in
`training/data/`, or supply per-tier corpus paths:

```bash
chmod +x train_all.sh
./train_all.sh
```

The script no longer regenerates or overwrites demo data by default. To
explicitly run the tiny synthetic pipeline demo:

```bash
NOVYRIX_USE_DEMO_DATA=true ./train_all.sh
```

For larger corpora, set `NOVYRIX_DATASET_4_0`, `NOVYRIX_DATASET_5_0`, and
`NOVYRIX_DATASET_6_0` to JSONL or plain-text files you have permission to use.
The trainer optimizes assistant responses for chat examples, validates on held-
out examples, and keeps the checkpoint with the best validation loss. Existing
checkpoints are unchanged until you train a tier again.

The script trains three Novyrix text models separately from random
initialization. It does not download a base language model:

```text
Novyrix 4.0
Novyrix 5.0
Novyrix 6.0
```

Training time depends on your device and configured step count. The current
small example datasets are development data only; models trained on them will
not be reliable general-purpose or coding assistants. Supply substantially
larger, high-quality, legally usable data before expecting meaningful capability.

The scratch checkpoints are stored in:

```text
training/output/novyrix-tier-4.0-scratch
training/output/novyrix-tier-5.0-scratch
training/output/novyrix-tier-6.0-scratch
```

The training script also creates the tier-specific training data and configuration it needs.


After training, make sure `.env` points each Novyrix tier to its corresponding checkpoint:

```env
NOVYRIX_MODEL_4_0_CHECKPOINT=./training/output/novyrix-tier-4.0-scratch
NOVYRIX_MODEL_5_0_CHECKPOINT=./training/output/novyrix-tier-5.0-scratch
NOVYRIX_MODEL_6_0_CHECKPOINT=./training/output/novyrix-tier-6.0-scratch
```

## Start

Start the Novyrix API:

```bash
python3 apps/api/server.py
```

Open:

```text
http://localhost:3000
```

## Model tiers

### Novyrix 4.0

Low mode: designed to use the smallest/faster Novyrix configuration. The engine
caps its reply at 512 generated tokens.

### Novyrix 5.0

Medium mode: the balanced Novyrix configuration. The engine caps its reply at
1,024 generated tokens.

### Novyrix 6.0

High mode: the largest of the three Novyrix scratch configurations. The engine
caps its reply at 1,536 generated tokens. It has not yet been benchmarked as a
capable assistant; size alone does not establish quality.

These are resource/configuration tiers, not measured intelligence levels. The
existing legacy checkpoints completed about 5,000, 5,600, and 7,400 training
steps on filtered public/curated data. Their validation loss is not a measure of
answer correctness: they can still produce incoherent text. More useful
capability requires substantially more high-quality data and compute.


### Contribute
If Novyrix can’t answer a question, it may search the web and save the results in a local cache. You can turn cached results into training examples with a command. Review the examples and their sources before including them in training, then run the training script to update the checkpoint. Restart the API afterward and test the model.

Have an idea for a behavior change or another improvement? I’d be happy to create a branch for your contribution and credit you.

Thank you for supporting Novyrix and helping me build it!
