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
    def make_prompt(messages, context_length: int | None = None) -> str:
        parts = []
        for message in messages:
            role = str(message.get("role", "user")).strip().lower()
            if role not in {"system", "user", "assistant"}:
                role = "user"
            content = str(message.get("content", ""))
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        prompt = "\n".join(parts)
        if context_length is None or len(prompt.encode("utf-8")) + 1 <= context_length:
            return prompt

        system = next(
            (str(message.get("content", "")) for message in messages if message.get("role") == "system"),
            "",
        )
        latest_user = next(
            (str(message.get("content", "")) for message in reversed(messages) if message.get("role") == "user"),
            "",
        )

        def format_prompt(system_text: str, user_text: str) -> str:
            return (
                f"<|im_start|>system\n{system_text}<|im_end|>\n"
                f"<|im_start|>user\n{user_text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

        target_bytes = max(0, context_length - 1)  # reserve one token for BOS
        overhead = len(format_prompt("", "").encode("utf-8"))
        available = max(0, target_bytes - overhead)
        system_budget = min(len(system.encode("utf-8")), min(192, available // 4))
        user_budget = max(0, available - system_budget)

        def fit_utf8(text: str, budget: int) -> str:
            data = text.encode("utf-8")
            if len(data) <= budget:
                return text
            marker = b"...[omitted]..."
            if budget <= len(marker):
                return data[:budget].decode("utf-8", errors="ignore")
            keep = budget - len(marker)
            prefix_size = (keep * 2) // 3
            suffix_size = keep - prefix_size
            prefix = data[:prefix_size].decode("utf-8", errors="ignore")
            suffix = data[-suffix_size:].decode("utf-8", errors="ignore") if suffix_size else ""
            return prefix + marker.decode("ascii") + suffix

        return format_prompt(fit_utf8(system, system_budget), fit_utf8(latest_user, user_budget))

    def generate(self, checkpoint, messages, max_new_tokens=512, temperature=0.7):
        tokenizer, model, device = self.load_model(checkpoint)
        prompt = self.make_prompt(messages, context_length=model.config.context_length)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor(
            [[model.config.bos_id, *prompt_ids]], dtype=torch.long, device=device
        )
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            stop_sequences=[tokenizer.encode("<|im_end|>", add_special_tokens=False)],
        )
        answer_ids = generated[0, input_ids.shape[1]:].tolist()
        answer = tokenizer.decode(answer_ids)
        # The training corpus represents turn boundaries as readable strings,
        # so trim them after decoding as well as stopping generation on them.
        for marker in ("<|im_end|>", "<|im_start|>"):
            answer = answer.split(marker, 1)[0]
        return answer.strip()

    def stream(self, checkpoint, messages, max_new_tokens=512, temperature=0.7):
        """Yield response text in chunks; generation currently completes before yielding."""
        text = self.generate(
            checkpoint, messages, max_new_tokens=max_new_tokens, temperature=temperature
        )
        for start in range(0, len(text), 24):
            yield text[start:start + 24]


runtime = ModelRuntime()
