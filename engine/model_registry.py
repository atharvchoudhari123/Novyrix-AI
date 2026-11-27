import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REGISTRY_FILE = ROOT / "models" / "registry.json"

SCRATCH_CHECKPOINTS = {
    "novyrix-4.0": ROOT / "training" / "output" / "novyrix-tier-4.0-scratch",
    "novyrix-5.0": ROOT / "training" / "output" / "novyrix-tier-5.0-scratch",
    "novyrix-6.0": ROOT / "training" / "output" / "novyrix-tier-6.0-scratch",
}

LEGACY_CHECKPOINTS = {
    "novyrix-4.0": ROOT / "training" / "output" / "novyrix-3.2-scratch",
    "novyrix-5.0": ROOT / "training" / "output" / "novyrix-4.0-scratch",
    "novyrix-6.0": ROOT / "training" / "output" / "novyrix-5.7-scratch",
}

LEGACY_CHECKPOINT_ENVS = {
    "novyrix-4.0": "NOVYRIX_3_2_CHECKPOINT",
    "novyrix-5.0": "NOVYRIX_4_0_CHECKPOINT",
    "novyrix-6.0": "NOVYRIX_5_7_CHECKPOINT",
}


def normalize_model_id(model_id):
    """Normalize unique legacy IDs; 4.0 is retained as the new canonical tier."""
    value = (model_id or "").strip().lower()

    value = {
        "novyrix-3.2": "novyrix-4.0",
        "novyrix-5.7": "novyrix-6.0",
    }.get(value, value)

    if value.startswith("novyrix-"):
        return "novyrix-" + value[len("novyrix-"):]

    return value


def get_models():
    if not REGISTRY_FILE.exists():
        return [
            {
                "id": "novyrix-4.0",
                "display_name": "Novyrix 4.0",
                "tier": "low",
                "max_new_tokens": 512,
                "temperature": 0.6,
                "checkpoint_env": "NOVYRIX_MODEL_4_0_CHECKPOINT",
            },
            {
                "id": "novyrix-5.0",
                "display_name": "Novyrix 5.0",
                "tier": "medium",
                "max_new_tokens": 1024,
                "temperature": 0.65,
                "checkpoint_env": "NOVYRIX_MODEL_5_0_CHECKPOINT",
            },
            {
                "id": "novyrix-6.0",
                "display_name": "Novyrix 6.0",
                "tier": "high",
                "max_new_tokens": 1536,
                "temperature": 0.65,
                "checkpoint_env": "NOVYRIX_MODEL_6_0_CHECKPOINT",
            },
        ]

    data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    return data.get("models", [])


def get_model(model_id):
    canonical = normalize_model_id(model_id)

    for model in get_models():
        if model["id"] == canonical:
            return model

    return None


def get_checkpoint(model_id):
    model = get_model(model_id)

    if model is None:
        raise ValueError(
            f"Unknown Novyrix model: {model_id}"
        )

    env_name = model.get("checkpoint_env", "")
    configured = os.getenv(env_name, "").strip() if env_name else ""
    if not configured:
        configured = os.getenv(LEGACY_CHECKPOINT_ENVS.get(model["id"], ""), "").strip()
    checkpoint = Path(configured).expanduser() if configured else SCRATCH_CHECKPOINTS[model["id"]]
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checkpoint = checkpoint.resolve()

    if not configured and not checkpoint.exists():
        legacy_checkpoint = LEGACY_CHECKPOINTS[model["id"]].resolve()
        if legacy_checkpoint.exists():
            checkpoint = legacy_checkpoint

    if not checkpoint.exists():
        return str(checkpoint)
    if not checkpoint.is_dir() and checkpoint.suffix != ".pt":
        raise ValueError(
            f"Novyrix only loads its own scratch checkpoints (.pt) or their output directories: {checkpoint}"
        )
    return str(checkpoint)
