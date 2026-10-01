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

These are the Novyrix product tiers.

The actual model checkpoints are configured separately.

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

The script trains the three Novyrix tiers separately:

```text
Novyrix 3.2
Novyrix 4.0
Novyrix 5.7
```

The script may take hours depending on internet speeds and HF downloads


The resulting checkpoints are stored in:

```text
training/output/novyrix-3.2
training/output/novyrix-4.0
training/output/novyrix-5.7
```

The training script also creates the tier-specific training data and configuration it needs.


After training, make sure `.env` points each Novyrix tier to its corresponding checkpoint:

```env
NOVYRIX_3_2_CHECKPOINT=./training/output/novyrix-3.2
NOVYRIX_4_0_CHECKPOINT=./training/output/novyrix-4.0
NOVYRIX_5_7_CHECKPOINT=./training/output/novyrix-5.7
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

Designed as the highest-capability Novyrix tier, with the largest practical model and most extensive training configuration.

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

- Base model
- Training data
- Training configuration
- Available compute
- Fine-tuning method
- Model size

Training the models does not automatically create a frontier-scale AI model. Larger and higher-quality datasets and appropriate compute are required to substantially improve model capabilities.

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
