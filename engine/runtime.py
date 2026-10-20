from __future__ import annotations

from pathlib import Path
import threading

import torch

from training.scratch_lm import ByteTokenizer, ModelConfig, NovyrixScratchLM


class ModelRuntime:
    """Inference runtime for Novyrix-owned, scratch-trained checkpoints only."""

    def __init__(self):
        self.cache = {}
        self.cache_lock = threading.Lock()

    def get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load_model(self, checkpoint: str):
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "checkpoint.pt"
        cache_key = str(checkpoint_path)
        with self.cache_lock:
            if cache_key in self.cache:
                return self.cache[cache_key]

            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"Novyrix scratch checkpoint not found: {checkpoint_path}. "
                    "Train this tier with the Novyrix from-scratch pipeline first."
                )

            device = self.get_device()
            print(f"Loading Novyrix scratch model on {device}: {checkpoint_path}")
            try:
                payload = torch.load(
                    checkpoint_path, map_location=device, weights_only=True
                )
                if payload.get("provenance") != "random initialization; no pretrained weights":
                    raise ValueError("Checkpoint provenance is not marked as scratch-trained.")
                config = ModelConfig(**payload["model_config"])
                model = NovyrixScratchLM(config).to(device)
                model.load_state_dict(payload["model_state_dict"], strict=True)
                model.eval()
            except Exception as error:
                raise RuntimeError(
                    f"Could not load Novyrix scratch checkpoint {checkpoint_path}: {error}"
                ) from error

            loaded = (ByteTokenizer(), model, device)
            self.cache[cache_key] = loaded
            return loaded

    @staticmethod
    def make_prompt(messages) -> str:
        parts = []
        for message in messages:
            role = str(message.get("role", "user")).upper()
            content = message.get("content", "")
            parts.append(f"{role}: {content}")
        parts.append("ASSISTANT:")
        return "\n\n".join(parts)

    def generate(self, checkpoint, messages, max_new_tokens=512, temperature=0.7):
        tokenizer, model, device = self.load_model(checkpoint)
        prompt = self.make_prompt(messages)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor(
            [[model.config.bos_id, *prompt_ids]], dtype=torch.long, device=device
        )
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        answer_ids = generated[0, input_ids.shape[1]:].tolist()
        return tokenizer.decode(answer_ids).strip()

    def stream(self, checkpoint, messages, max_new_tokens=512, temperature=0.7):
        """Yield response text in chunks; generation currently completes before yielding."""
        text = self.generate(
            checkpoint, messages, max_new_tokens=max_new_tokens, temperature=temperature
        )
        for start in range(0, len(text), 24):
            yield text[start:start + 24]


runtime = ModelRuntime()
