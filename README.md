[Docs](apps/web/index.html)

# Novyrix

Novyrix is a self-hosted AI platform.

## Model ownership and current status

**The current Novyrix language model is not a foundation model trained from scratch by the Novyrix project.** The default checkpoint in this repository is `Qwen/Qwen2.5-0.5B-Instruct`. The Novyrix model names and tiers are product labels; they do not mean that each tier is a separately invented or independently pretrained model architecture.

The repository's current training pipeline fine-tunes existing pretrained checkpoints on Novyrix data. Fine-tuning changes model weights, but it does not make the original base model disappear or make the result a from-scratch pretrained foundation model. Unless a deployment explicitly points at a different checkpoint, the default is the Qwen checkpoint above.

This distinction matters for capability claims: planning prompts, tools, branding, and fine-tuning can improve a product, but they do not guarantee reliable coding or frontier-level reasoning. The current model may make mistakes, including confident factual errors. Please evaluate outputs before relying on them.

The `novyrix-from-scratch` branch is the experimental effort to build a Novyrix-owned model from randomly initialized weights, without loading Qwen or another pretrained language model. Until that model has actually been trained and evaluated, it is a development project—not a claim that a capable independent Novyrix foundation model already exists. See [the from-scratch model design](docs/own-model-design.md).

## Platform components

It contains:

- Novyrix API
- Novyrix Engine and agent/tool orchestration
- Model registry and runtime
- Training pipeline
- File handling
- Web interface and Coding Playground
- Docker configuration

## Current model tiers

Novyrix 3.2, 4.0, and 5.7 are product tiers. They are not proof of model size, independent architecture, or training from scratch. The actual checkpoint for each tier is configured separately; see `models/registry.json` and the `NOVYRIX_*_CHECKPOINT` environment variables.

## Architecture

```text
Browser
    |
Novyrix API
    |
Novyrix Engine
    +-- Memory and context
    +-- File handling
    +-- Model routing
    +-- Agent/tool orchestration
    |
Model runtime
    |
Configured checkpoint (currently Qwen by default)
```

The agent engine can classify requests, plan tasks, and call selected tools. It is an orchestration layer around a language model, not itself a pretrained language model.

## Install

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train the current fine-tuned tiers

Run:

```bash
chmod +x train_all.sh
./train_all.sh
```

This current pipeline trains the three Novyrix product tiers using the configured pretrained base checkpoints and tier-specific data. It is a fine-tuning workflow, not pretraining from random initialization.

The resulting checkpoints are stored in:

```text
training/output/novyrix-3.2
training/output/novyrix-4.0
training/output/novyrix-5.7
```

After training, configure `.env` to point at the resulting checkpoints:

```env
NOVYRIX_3_2_CHECKPOINT=./training/output/novyrix-3.2
NOVYRIX_4_0_CHECKPOINT=./training/output/novyrix-4.0
NOVYRIX_5_7_CHECKPOINT=./training/output/novyrix-5.7
```

Training duration and achievable quality depend on the base model, training data, method, and available compute.

## Start

```bash
python3 apps/api/server.py
```

Open `http://localhost:3000`.

## Capabilities and limitations

The platform is designed to support chat, coding, debugging, file analysis, website generation, tool/plugin integration, defensive security review, image/video generation, and model training. These are software features, not guarantees that the configured language model will perform every task correctly.

Training on a small dataset or adding a detailed prompt does not by itself create a strong coding model. A genuinely from-scratch model needs its own tokenizer and randomly initialized architecture, substantial suitable pretraining data and compute, followed by instruction tuning and honest evaluation. The initial from-scratch model may be much less capable than Qwen; capability must be demonstrated, not inferred from the Novyrix name.

## Tokens and memberships

The API supports 100 daily tokens by default, with each chat message costing 5 tokens. Daily tokens reset by UTC date. Purchased credits are separate from the daily allowance.

Membership access is enforced by the API:

- Free: Novyrix 3.2
- Core: Novyrix 3.2 + 4.0
- Premium: Novyrix 3.2 + 4.0 + 5.7

The account endpoints are `/v1/account`, `/v1/account/credits`, and `/v1/account/membership`. Credits and membership write endpoints are placeholders for future Stripe Checkout/webhook integration; they do not verify payment.
