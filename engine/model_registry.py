import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REGISTRY_FILE = ROOT / "models" / "registry.json"

SCRATCH_CHECKPOINTS = {
    "novyrix-3.2": ROOT / "training" / "output" / "novyrix-3.2-scratch",
    "novyrix-4.0": ROOT / "training" / "output" / "novyrix-4.0-scratch",
    "novyrix-5.7": ROOT / "training" / "output" / "novyrix-5.7-scratch",
}


def normalize_model_id(model_id):
    """Normalize legacy Novyrix IDs to canonical Novyrix IDs."""
    value = (model_id or "").strip().lower()

    if value.startswith("novyrix-"):
        return "novyrix-" + value[len("novyrix-"):]

    return value


def get_models():
    if not REGISTRY_FILE.exists():
        return [
            {
                "id": "novyrix-3.2",
                "display_name": "Novyrix 3.2",
                "tier": "fast",
                "checkpoint_env": "NOVYRIX_3_2_CHECKPOINT",
            },
            {
                "id": "novyrix-4.0",
                "display_name": "Novyrix 4.0",
                "tier": "general",
                "checkpoint_env": "NOVYRIX_4_0_CHECKPOINT",
            },
            {
                "id": "novyrix-5.7",
                "display_name": "Novyrix 5.7",
                "tier": "advanced",
                "checkpoint_env": "NOVYRIX_5_7_CHECKPOINT",
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
    checkpoint = Path(configured).expanduser() if configured else SCRATCH_CHECKPOINTS[model["id"]]
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    checkpoint = checkpoint.resolve()

    if not checkpoint.exists():
        return str(checkpoint)
    if not checkpoint.is_dir() and checkpoint.suffix != ".pt":
        raise ValueError(
            f"Novyrix only loads its own scratch checkpoints (.pt) or their output directories: {checkpoint}"
        )
    return str(checkpoint)
