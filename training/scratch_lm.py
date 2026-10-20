"""Train a small Novyrix decoder-only language model from random initialization.

This module deliberately does not import Transformers or load a pretrained model.
The tokenizer is a transparent UTF-8 byte codec (256 byte IDs plus BOS/EOS/PAD).
It is a Stage-1 ownership/pipeline baseline, not a capable general-purpose model.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 259
    context_length: int = 256
    width: int = 256
    layers: int = 4
    heads: int = 4
    dropout: float = 0.0
    bos_id: int = 0
    eos_id: int = 1
    pad_id: int = 2


class ByteTokenizer:
    """Reversible UTF-8 byte tokenizer; IDs 3..258 represent byte values 0..255."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        tokens = [byte + 3 for byte in text.encode("utf-8")]
        return [0, *tokens, 1] if add_special_tokens else tokens

    def decode(self, token_ids: list[int]) -> str:
        raw = bytes(token - 3 for token in token_ids if 3 <= token <= 258)
        return raw.decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "type": "novyrix_utf8_byte_v1",
                    "vocab_size": 259,
                    "special_tokens": {"bos": 0, "eos": 1, "pad": 2},
                    "byte_id_offset": 3,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.width % config.heads:
            raise ValueError("width must be divisible by heads")
        self.heads = config.heads
        self.head_width = config.width // config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.output = nn.Linear(config.width, config.width, bias=False)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).view(
            batch, length, 3, self.heads, self.head_width
        ).permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, width)
        return self.output(attended)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.attention = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x)))
        return x + self.dropout(self.mlp(self.norm2(x)))


class NovyrixScratchLM(nn.Module):
    """Compact decoder-only Transformer initialized entirely from scratch."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.context_length, config.width)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.layers)
        )
        self.final_norm = nn.LayerNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, length = input_ids.shape
        if length > self.config.context_length:
            raise ValueError("input exceeds configured context_length")
        positions = torch.arange(length, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)[None]
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=self.config.pad_id,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.context_length :]
            logits, _ = self(context)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                cutoff = torch.topk(next_logits, min(top_k, next_logits.size(-1))).values[:, -1, None]
                next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))
            next_id = torch.multinomial(torch.softmax(next_logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
            if torch.all(next_id == self.config.eos_id):
                break
        return input_ids


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_corpus(paths: list[Path], tokenizer: ByteTokenizer) -> torch.Tensor:
    ids: list[int] = []
    for path in paths:
        if path.suffix.lower() == ".jsonl":
            documents = []
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
                text = record.get("text")
                if not isinstance(text, str):
                    messages = record.get("messages", [])
                    text = "\n".join(
                        str(message.get("content", ""))
                        for message in messages
                        if isinstance(message, dict)
                    )
                if text:
                    documents.append(text)
            text = "\n\n".join(documents)
        else:
            text = path.read_text(encoding="utf-8")
        ids.extend(tokenizer.encode(text))
    return torch.tensor(ids, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", nargs="+", type=Path, required=True,
                        help="UTF-8 text files you have permission to use.")
    parser.add_argument("--output-dir", type=Path, default=Path("training/output/novyrix-scratch"))
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.steps < 1 or args.eval_every < 1 or args.batch_size < 1:
        parser.error("steps, eval-every, and batch-size must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    config = ModelConfig(
        context_length=args.context_length, width=args.width,
        layers=args.layers, heads=args.heads,
    )
    tokenizer = ByteTokenizer()
    tokens = load_corpus(args.corpus, tokenizer)
    split = int(len(tokens) * 0.98)
    train_tokens, val_tokens = tokens[:split], tokens[split:]
    minimum = config.context_length + 2
    if train_tokens.numel() < minimum or val_tokens.numel() < minimum:
        parser.error(
            f"corpus is too small: need at least {2 * minimum} encoded tokens "
            "for a 98/2 train-validation split"
        )
    train_tokens, val_tokens = train_tokens.to(device), val_tokens.to(device)

    model = NovyrixScratchLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.output_dir / "tokenizer.json")
    (args.output_dir / "model_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    corpus_manifest = [
        {"path": str(path), "bytes": path.stat().st_size}
        for path in args.corpus
    ]
    (args.output_dir / "training_manifest.json").write_text(
        json.dumps(
            {"provenance": "random initialization; no pretrained weights",
             "seed": args.seed, "steps": args.steps,
             "corpus": corpus_manifest},
            indent=2,
        ),
        encoding="utf-8",
    )

    def batch(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        starts = torch.randint(
            0, source.numel() - config.context_length - 1, (args.batch_size,),
            device=device,
        )
        x = torch.stack([source[i:i + config.context_length] for i in starts])
        y = torch.stack([source[i + 1:i + config.context_length + 1] for i in starts])
        return x, y

    @torch.no_grad()
    def estimate_loss() -> dict[str, float]:
        model.eval()
        result = {}
        for name, source in (("train", train_tokens), ("validation", val_tokens)):
            losses = []
            for _ in range(args.eval_batches):
                x, y = batch(source)
                _, loss = model(x, y)
                losses.append(float(loss))
            result[name] = sum(losses) / len(losses)
        model.train()
        return result

    for step in range(1, args.steps + 1):
        model.train()
        x, y = batch(train_tokens)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            losses = estimate_loss()
            print(
                f"step={step} train_loss={losses['train']:.4f} "
                f"validation_loss={losses['validation']:.4f} "
                f"validation_perplexity={math.exp(min(losses['validation'], 20)):.2f}",
                flush=True,
            )
            torch.save(
                {"model_config": asdict(config),
                 "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "step": step, "seed": args.seed,
                 "provenance": "random initialization; no pretrained weights"},
                args.output_dir / "checkpoint.pt",
            )


if __name__ == "__main__":
    main()

