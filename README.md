[Docs](apps/web/index.html)

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

## Capabilities

Novyrix is designed to support:

- Chat
- Multi-turn conversations
- Coding
- Debugging
- File analysis
- Website generation
- Plugin/tool integration
- Defensive security review
- Image generation
- Video generation
- Custom model training
- Multiple model tiers

## Important

The quality of each Novyrix model depends on:

- Training data
- Training configuration
- Available compute
- Evaluation and iteration

Training uses pinned, permissively licensed external examples by default
(Apache-2.0 general conversations and MIT Python instructions) alongside
Novyrix examples. Training remains from random initialization—no third-party
model weights are loaded. Tier budgets scale from 5k to 24k steps, with more
coding data for 5.0 and the broadest mix for 6.0. These are still small scratch
models, not assistants that can answer anything. See `training/README.md` for
source details, overrides, and safe retraining steps.

The training code and initialized weights are Novyrix-owned. The current small
datasets are not sufficient to produce strong models. Do not infer coding or
factual reliability from tier names; evaluate each trained checkpoint.

## Agent engine

Novyrix includes a shared agent/tool layer for all model tiers. The current deterministic router can use the calculator, workspace file tools, debugging helpers, tests, and selective web research. Model selection changes the scratch checkpoint and generation budget; it does not grant broader tool permissions.

In chat mode, the runtime handles simple greetings directly and suppresses
obviously corrupted model output with a readable warning. This guard cannot
verify factual accuracy or turn an undertrained checkpoint into a capable model.

Workspace write tools are restricted to the Novyrix project root. Web tools fetch HTTP(S) context with explicit timeouts.

Factual lookup questions (for example, “list Tesla models” or “what year did
this happen?”) trigger web search because a small scratch model cannot reliably
recognize when it is uncertain. Research replies show attributed search snippets
and source links directly rather than asking the model to paraphrase them.
Successful structured results are cached locally in
`data/research_memory.json` (override with `NOVYRIX_KNOWLEDGE_PATH`) and can be
reused for closely matching questions for up to 30 days. Failed searches and raw
web pages are not saved. This is retrieval memory, not training: it does not
change model weights or add examples to the training dataset. Search snippets
are not independent verification; follow the links for context and check
time-sensitive claims.

## Tokens and memberships

The API now supports 100 daily tokens by default, with each chat message costing 5 tokens. Daily tokens reset by UTC date. Purchased credits are separate from the daily allowance.

Membership access is enforced by the API:

- Free: Novyrix 4.0
- Core: Novyrix 4.0 + 5.0
- Premium: Novyrix 4.0 + 5.0 + 6.0

The account endpoints are `/v1/account`, `/v1/account/credits`, and `/v1/account/membership`. The credits and membership write endpoints are placeholders for future Stripe Checkout/webhook integration; they are not payment verification.

### Local developer mode

The Developer Mode button unlocks all three Novyrix text tiers without
membership or token deductions for a 12-hour session. The server validates the
password and sets an HttpOnly session cookie; the password is not shipped to
the browser. Configure `NOVYRIX_DEV_MODE_PASSWORD` in your private `.env` file
and never commit that file. This is a local development convenience, not a
production authentication system.
# Novyrix-AI


## Novyrix Advanced beta

The `novrix-advanced-beta` branch adds an experimental execution layer focused on
making Novyrix feel like its own capable assistant rather than copying another
assistant's personality.

Advanced beta adds:

- broad request classification instead of a small coding keyword list
- explicit task planning before generation
- multi-tool routing when a task benefits from tools
- coding/build/debug/research/file/math workflows
- tool evidence injection so the model can reason from actual results
- the same Novyrix membership and token enforcement
- a clear distinction between what Novyrix completed and what it could not execute

The planner is deliberately deterministic and transparent. It does not pretend
that a small local model has frontier-level reasoning. The model remains the
language-generation layer, while the Advanced planner and tool system provide
structure and execution support.

For local testing, enable:

```env
NOVYRIX_ADVANCED_BETA=true
```

The experimental branch does not change `main`.
