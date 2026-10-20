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

Novyrix 3.2  
Novyrix 4.0  
Novyrix 5.7

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

Run the unified training script:

```bash
chmod +x train_all.sh
./train_all.sh
```

The script trains three Novyrix text models separately from random
initialization. It does not download a base language model:

```text
Novyrix 3.2
Novyrix 4.0
Novyrix 5.7
```

Training time depends on your device and configured step count. The current
small example datasets are development data only; models trained on them will
not be reliable general-purpose or coding assistants. Supply substantially
larger, high-quality, legally usable data before expecting meaningful capability.

The scratch checkpoints are stored in:

```text
training/output/novyrix-3.2-scratch
training/output/novyrix-4.0-scratch
training/output/novyrix-5.7-scratch
```

The training script also creates the tier-specific training data and configuration it needs.


After training, make sure `.env` points each Novyrix tier to its corresponding checkpoint:

```env
NOVYRIX_3_2_CHECKPOINT=./training/output/novyrix-3.2-scratch
NOVYRIX_4_0_CHECKPOINT=./training/output/novyrix-4.0-scratch
NOVYRIX_5_7_CHECKPOINT=./training/output/novyrix-5.7-scratch
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

### Novyrix 3.2

Designed as the faster, lightweight Novyrix tier.

### Novyrix 4.0

Designed as the balanced general-purpose Novyrix tier.

### Novyrix 5.7

The largest of the three Novyrix scratch configurations. It has not yet been
benchmarked as a capable assistant; size alone does not establish quality.

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

The training code and initialized weights are Novyrix-owned. The current small
datasets are not sufficient to produce strong models. Do not infer coding or
factual reliability from tier names; evaluate each trained checkpoint.

## Agent engine

Novyrix now includes a shared agent/tool layer for all model tiers. The tool ceiling differs by tier:

- Novyrix 3.2: calculator and basic workspace file access.
- Novyrix 4.0: workspace writing/editing, file search, debugging, tests, and web context tools.
- Novyrix 5.7: the same tool surface with the highest model tier available to the deployment.

Workspace write tools are restricted to the Novyrix project root. Web tools fetch HTTP(S) context with explicit timeouts.

## Tokens and memberships

The API now supports 100 daily tokens by default, with each chat message costing 5 tokens. Daily tokens reset by UTC date. Purchased credits are separate from the daily allowance.

Membership access is enforced by the API:

- Free: Novyrix 3.2
- Core: Novyrix 3.2 + 4.0
- Premium: Novyrix 3.2 + 4.0 + 5.7

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
