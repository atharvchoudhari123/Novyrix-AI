[Docs](apps/web/index.html)

# LumaCore

LumaCore is a self-hosted AI platform.

It contains:

- LumaCore API
- LumaCore Engine
- Model registry
- Model runtime
- Training pipeline
- File handling
- Web interface
- Coding Playground
- Docker configuration

## LumaCore models

LumaCore 3.2  
LumaCore 4.0  
LumaCore 5.7

These are the LumaCore product tiers.

The actual model checkpoints are configured separately.

## Architecture

```text
Browser
    |
    v
LumaCore API
    |
    v
LumaCore Engine
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
LumaCore Runtime
    |
    v
LumaCore Model
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

## Train all LumaCore model tiers

Run the unified training script:

```bash
chmod +x train_all.sh
./train_all.sh
```

The script trains the three LumaCore tiers separately:

```text
LumaCore 3.2
LumaCore 4.0
LumaCore 5.7
```

The script may take hours depending on internet speeds and HF downloads


The resulting checkpoints are stored in:

```text
training/output/lumacore-3.2
training/output/lumacore-4.0
training/output/lumacore-5.7
```

The training script also creates the tier-specific training data and configuration it needs.


After training, make sure `.env` points each LumaCore tier to its corresponding checkpoint:

```env
LUMACORE_3_2_CHECKPOINT=./training/output/lumacore-3.2
LUMACORE_4_0_CHECKPOINT=./training/output/lumacore-4.0
LUMACORE_5_7_CHECKPOINT=./training/output/lumacore-5.7
```

## Start

Start the LumaCore API:

```bash
python3 apps/api/server.py
```

Open:

```text
http://localhost:3000
```

## Model tiers

### LumaCore 3.2

Designed as the faster, lightweight LumaCore tier.

### LumaCore 4.0

Designed as the balanced general-purpose LumaCore tier.

### LumaCore 5.7

Designed as the highest-capability LumaCore tier, with the largest practical model and most extensive training configuration.

## Capabilities

LumaCore is designed to support:

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

The quality of each LumaCore model depends on:

- Base model
- Training data
- Training configuration
- Available compute
- Fine-tuning method
- Model size

Training the models does not automatically create a frontier-scale AI model. Larger and higher-quality datasets and appropriate compute are required to substantially improve model capabilities.

## Agent engine

LumaCore now includes a shared agent/tool layer for all model tiers. The tool ceiling differs by tier:

- LumaCore 3.2: calculator and basic workspace file access.
- LumaCore 4.0: workspace writing/editing, file search, debugging, tests, and web context tools.
- LumaCore 5.7: the same tool surface with the highest model tier available to the deployment.

Workspace write tools are restricted to the LumaCore project root. Web tools fetch HTTP(S) context with explicit timeouts.

## Tokens and memberships

The API now supports 100 daily tokens by default, with each chat message costing 5 tokens. Daily tokens reset by UTC date. Purchased credits are separate from the daily allowance.

Membership access is enforced by the API:

- Free: LumaCore 3.2
- Core: LumaCore 3.2 + 4.0
- Premium: LumaCore 3.2 + 4.0 + 5.7

The account endpoints are `/v1/account`, `/v1/account/credits`, and `/v1/account/membership`. The credits and membership write endpoints are placeholders for future Stripe Checkout/webhook integration; they are not payment verification.
