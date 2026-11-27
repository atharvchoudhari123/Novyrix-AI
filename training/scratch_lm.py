"""Train a small Novyrix decoder-only language model from random initialization.

This module deliberately does not import Transformers or load a pretrained model.
The tokenizer is a transparent UTF-8 byte codec (256 byte IDs plus BOS/EOS/PAD).
It is a Stage-1 ownership/pipeline baseline, not a capable general-purpose model.
"""
from __future__ import annotations

import argparse
from array import array
import json
import math
import os
import random
import re
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

    bos_id = 0
    eos_id = 1
    pad_id = 2

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


@dataclass
class TrainingExample:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    group_key: str = ""


class IndexedJsonlSource:
    """Disk-backed random sampler for large JSONL corpora.

    Only byte offsets and validation flags stay in RAM; example text is read
    from disk as needed, so multi-million-row corpora do not become millions
    of Python/PyTorch objects in memory.
    """

    def __init__(self, path: Path, validation_fraction: float, seed: int):
        self.path = path
        self.offsets = array("Q")
        self.validation_flags = bytearray()
        self.handle = path.open("rb")
        chooser = random.Random(seed)
        with path.open("rb") as source:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                self.offsets.append(offset)
                self.validation_flags.append(1 if chooser.random() < validation_fraction else 0)
        self.validation_count = sum(self.validation_flags)
        self.training_count = len(self.offsets) - self.validation_count
        if not self.training_count or not self.validation_count:
            self.handle.close()
            raise ValueError(f"Large corpus needs both training and validation rows: {path}")

    def sample(self, split: str, tokenizer: ByteTokenizer, random_crop: bool = True) -> TrainingExample:
        validation = split == "validation"
        count = self.validation_count if validation else self.training_count
        if count <= 0:
            raise ValueError(f"No {split} examples available in {self.path}")
        while True:
            index = random.randrange(len(self.offsets))
            if bool(self.validation_flags[index]) != validation:
                continue
            self.handle.seek(self.offsets[index])
            record = json.loads(self.handle.readline())
            text, spans = _record_to_text_and_spans(record, self.path, index + 1)
            example = _encode_training_example(text, spans, tokenizer)
            if example is not None:
                example.group_key = str(record.get("id") or index)
                return example


class TrainingPool:
    """Mix small in-memory instruction data with large disk-backed text."""

    def __init__(
        self, examples, large_sources, split: str, tokenizer: ByteTokenizer,
        instruction_fraction: float = 0.65,
    ):
        self.examples = examples
        self.large_sources = large_sources
        self.split = split
        self.tokenizer = tokenizer
        self.instruction_fraction = instruction_fraction
        self.total_count = len(examples) + sum(
            source.validation_count if split == "validation" else source.training_count
            for source in large_sources
        )

    def sample(self, random_crop: bool = True) -> TrainingExample:
        if self.large_sources and self.examples:
            # Web corpora add language exposure, but should not drown out the
            # supervised examples that teach the model to answer prompts.
            use_instruction_data = random.random() < self.instruction_fraction
        else:
            use_instruction_data = bool(self.examples)
        if use_instruction_data:
            return random.choice(self.examples)
        weights = [
            source.validation_count if self.split == "validation" else source.training_count
            for source in self.large_sources
        ]
        source = random.choices(self.large_sources, weights=weights, k=1)[0]
        return source.sample(self.split, self.tokenizer, random_crop=random_crop)


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
        stop_sequences: list[list[int]] | None = None,
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
            if stop_sequences and any(
                len(sequence) <= input_ids.shape[1]
                and input_ids[0, -len(sequence):].tolist() == sequence
                for sequence in stop_sequences
            ):
                break
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


def _chat_text(messages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """Serialize turns and return character spans that should receive loss."""
    parts: list[str] = []
    assistant_spans: list[tuple[int, int]] = []
    length = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "user")).strip().lower()
        if role not in {"system", "user", "assistant"}:
            continue
        content = str(message.get("content", ""))
        header = f"<|im_start|>{role}\n"
        parts.append(header)
        length += len(header)
        content_start = length
        parts.append(content)
        length += len(content)
        parts.append("<|im_end|>\n")
        length += len("<|im_end|>\n")
        if role == "assistant":
            # Teach the model to end the turn, but do not train it to imitate
            # system or user text.
            assistant_spans.append((content_start, length - 1))
    return "".join(parts), assistant_spans


def _record_to_text_and_spans(record: dict, path: Path, line_number: int):
    messages = record.get("messages")
    if isinstance(messages, list):
        return _chat_text(messages)

    prompt, response = record.get("prompt"), record.get("response")
    if isinstance(prompt, str) and isinstance(response, str):
        return _chat_text([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ])

    text = record.get("text")
    if not isinstance(text, str):
        raise ValueError(f"Expected text, messages, or prompt/response at {path}:{line_number}")

    # Current scratch datasets store a serialized chat in `text`. Mask every
    # non-assistant span so optimization is focused on answers.
    spans = [
        (match.start(1), match.end(0))
        for match in re.finditer(
            r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", text, re.DOTALL
        )
    ]
    return text, spans


def _encode_training_example(
    text: str,
    assistant_spans: list[tuple[int, int]],
    tokenizer: ByteTokenizer,
) -> TrainingExample | None:
    encoded = tokenizer.encode(text)
    targets = [tokenizer.pad_id] * (len(encoded) - 1)
    if not assistant_spans:
        targets = encoded[1:]
    else:
        # Byte-token indices align with UTF-8 byte offsets, not Python chars.
        byte_spans = [
            (len(text[:start].encode("utf-8")), len(text[:end].encode("utf-8")))
            for start, end in assistant_spans
        ]
        for target_index, token_id in enumerate(encoded[1:]):
            if any(start <= target_index < end for start, end in byte_spans):
                targets[target_index] = token_id
        if byte_spans and byte_spans[-1][1] == len(text.encode("utf-8")):
            targets[-1] = encoded[-1]  # EOS after a final assistant turn.

    if not any(token != tokenizer.pad_id for token in targets):
        return None
    return TrainingExample(
        input_ids=torch.tensor(encoded[:-1], dtype=torch.long),
        target_ids=torch.tensor(targets, dtype=torch.long),
    )


def load_training_examples(
    paths: list[Path], tokenizer: ByteTokenizer, stats: dict | None = None,
    validation_fraction: float = 0.15, seed: int = 42,
) -> tuple[list[TrainingExample], list[IndexedJsonlSource]]:
    examples: list[TrainingExample] = []
    large_sources: list[IndexedJsonlSource] = []
    seen_records: set[str] = set()
    duplicates = 0

    def add_example(text: str, spans: list[tuple[int, int]], path: Path, line_number: int):
        nonlocal duplicates
        exact_key = " ".join(text.split())
        if exact_key in seen_records:
            duplicates += 1
            return
        seen_records.add(exact_key)
        prompts = re.findall(
            r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, re.DOTALL
        )
        prompt = " ".join(" ".join(prompts).casefold().split())
        prompt = re.sub(r"[^\w\s]", "", prompt)
        example = _encode_training_example(text, spans, tokenizer)
        if example is not None:
            example.group_key = prompt or exact_key.casefold()
            examples.append(example)

    for path in paths:
        if path.suffix.lower() == ".jsonl":
            if path.stat().st_size >= 128 * 1024 * 1024:
                large_sources.append(
                    IndexedJsonlSource(path, validation_fraction, seed + len(large_sources))
                )
                continue
            with path.open("r", encoding="utf-8") as corpus:
                lines = enumerate(corpus, 1)
                for line_number, line in lines:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
                    text, spans = _record_to_text_and_spans(record, path, line_number)
                    add_example(text, spans, path, line_number)
        else:
            # Plain text corpora use next-token pretraining over paragraphs.
            with path.open("r", encoding="utf-8") as corpus:
                for paragraph in re.split(r"\n\s*\n", corpus.read()):
                    if paragraph.strip():
                        add_example(paragraph, [], path, 0)
    if stats is not None:
        stats["duplicate_examples_removed"] = duplicates
        stats["large_jsonl_examples"] = sum(len(source.offsets) for source in large_sources)
    return examples, large_sources


def split_by_prompt_group(
    examples: list[TrainingExample], validation_fraction: float, seed: int
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    groups: dict[str, list[TrainingExample]] = {}
    for example in examples:
        groups.setdefault(example.group_key, []).append(example)
    if len(groups) < 2:
        raise ValueError("Need at least two distinct prompt groups for a leak-free validation split.")
    group_keys = list(groups)
    random.Random(seed + 1).shuffle(group_keys)
    validation_count = min(
        len(group_keys) - 1,
        max(1, round(len(group_keys) * validation_fraction)),
    )
    validation_groups = set(group_keys[:validation_count])
    train = [example for key, values in groups.items() if key not in validation_groups for example in values]
    validation = [example for key, values in groups.items() if key in validation_groups for example in values]
    return train, validation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", nargs="+", type=Path, required=True,
                        help="UTF-8 text files you have permission to use.")
    parser.add_argument("--output-dir", type=Path, default=Path("training/output/novyrix-scratch"))
    parser.add_argument("--source-manifest", type=Path, nargs="*", default=[],
                        help="Optional JSON manifests describing corpus sources and licenses.")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop after this many evaluations without improvement; 0 disables early stopping.")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="Minimum validation-loss improvement needed to reset patience.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--instruction-fraction", type=float,
        default=float(os.environ.get("NOVYRIX_INSTRUCTION_FRACTION", "0.65")),
        help="Share of batches sampled from chat/instruction examples when web text is also present.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.steps < 1 or args.eval_every < 1 or args.batch_size < 1:
        parser.error("steps, eval-every, and batch-size must be positive")
    if args.eval_batches < 1 or args.gradient_accumulation_steps < 1:
        parser.error("eval-batches and gradient-accumulation-steps must be positive")
    if args.early_stopping_patience < 0 or args.min_delta < 0:
        parser.error("early-stopping-patience and min-delta must be nonnegative")
    if args.warmup_steps < 0 or not 0 <= args.min_lr_ratio <= 1:
        parser.error("warmup-steps must be nonnegative and min-lr-ratio must be between 0 and 1")
    if not 0.05 <= args.validation_fraction <= 0.5:
        parser.error("validation-fraction must be between 0.05 and 0.5")
    if not 0.0 <= args.instruction_fraction <= 1.0:
        parser.error("instruction-fraction must be between 0 and 1")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    config = ModelConfig(
        context_length=args.context_length, width=args.width,
        layers=args.layers, heads=args.heads,
    )
    tokenizer = ByteTokenizer()
    data_stats: dict[str, int] = {}
    examples, large_sources = load_training_examples(
        args.corpus, tokenizer, stats=data_stats,
        validation_fraction=args.validation_fraction, seed=args.seed,
    )
    if len(examples) >= 2:
        train_memory, val_memory = split_by_prompt_group(
            examples, args.validation_fraction, args.seed
        )
    elif examples and large_sources:
        train_memory, val_memory = examples, []
    elif len(examples) < 2 and not large_sources:
        parser.error(
            "need at least two usable examples to create separate training and "
            "validation sets; add more licensed data or JSONL examples"
        )
    else:
        train_memory, val_memory = examples, []
    train_pool = TrainingPool(
        train_memory, large_sources, "train", tokenizer, args.instruction_fraction
    )
    val_pool = TrainingPool(
        val_memory, large_sources, "validation", tokenizer, args.instruction_fraction
    )
    train_count = train_pool.total_count
    validation_count = val_pool.total_count
    unique_count = len(examples) + data_stats.get("large_jsonl_examples", 0)
    prompt_groups = len({example.group_key for example in examples}) + data_stats.get(
        "large_jsonl_examples", 0
    )
    if prompt_groups < 100:
        print(
            f"WARNING: only {prompt_groups} unique prompt groups remain after deduplication; "
            "this is pipeline/demo-scale data, not broad capability training.",
            flush=True,
        )
    print(
        f"data_examples={unique_count} duplicates_removed={data_stats.get('duplicate_examples_removed', 0)} "
        f"disk_backed_examples={data_stats.get('large_jsonl_examples', 0)} "
        f"unique_prompt_groups={prompt_groups} train_examples={train_count} "
        f"validation_examples={validation_count}",
        flush=True,
    )

    model = NovyrixScratchLM(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(args.output_dir / "tokenizer.json")
    (args.output_dir / "model_config.json").write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )
    corpus_manifest = [
        {"path": str(path), "bytes": path.stat().st_size}
        for path in args.corpus
    ]
    source_manifest = []
    for path in args.source_manifest:
        if not path.is_file():
            parser.error(f"source manifest not found: {path}")
        source_manifest.append(json.loads(path.read_text(encoding="utf-8")))
    run_manifest = {
             "status": "in_progress",
             "provenance": "random initialization; no pretrained weights",
             "seed": args.seed, "steps": args.steps,
             "objective": "assistant-only loss for serialized chats; next-token loss for plain text",
             "training_examples": train_count,
             "validation_examples": validation_count,
             "unique_examples": unique_count,
             "duplicate_examples_removed": data_stats.get("duplicate_examples_removed", 0),
             "unique_prompt_groups": prompt_groups,
             "validation_fraction": args.validation_fraction,
             "instruction_fraction": args.instruction_fraction,
             "learning_rate": args.learning_rate,
             "weight_decay": args.weight_decay,
             "warmup_steps": args.warmup_steps,
             "min_lr_ratio": args.min_lr_ratio,
             "gradient_accumulation_steps": args.gradient_accumulation_steps,
             "early_stopping_patience": args.early_stopping_patience,
             "min_delta": args.min_delta,
             "corpus": corpus_manifest,
             "data_sources": source_manifest,
             "actual_steps": None,
             "best_step": None,
             "best_validation_loss": None,
    }
    manifest_path = args.output_dir / "training_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    def batch(source: TrainingPool, random_crop: bool = True):
        inputs, targets = [], []
        for _ in range(args.batch_size):
            example = source.sample(random_crop=random_crop)
            last_start = max(0, example.input_ids.numel() - config.context_length)
            candidate_starts = torch.arange(last_start + 1)
            active_prefix = F.pad(
                (example.target_ids != tokenizer.pad_id).to(torch.int32).cumsum(0),
                (1, 0),
            )
            window_ends = (candidate_starts + config.context_length).clamp(
                max=example.target_ids.numel()
            )
            has_targets = (
                active_prefix[window_ends] - active_prefix[candidate_starts]
            ) > 0
            starts = candidate_starts[has_targets].tolist()
            if not starts:
                raise ValueError("Training example has no supervised target in any context window")
            start = random.choice(starts) if random_crop else starts[0]
            x = example.input_ids[start:start + config.context_length]
            y = example.target_ids[start:start + config.context_length]
            padding = config.context_length - x.numel()
            if padding:
                x = F.pad(x, (0, padding), value=tokenizer.pad_id)
                y = F.pad(y, (0, padding), value=tokenizer.pad_id)
            inputs.append(x)
            targets.append(y)
        return torch.stack(inputs).to(device), torch.stack(targets).to(device)

    @torch.no_grad()
    def estimate_loss() -> dict[str, float]:
        model.eval()
        result = {}
        for name, source in (("train", train_pool), ("validation", val_pool)):
            losses = []
            for _ in range(args.eval_batches):
                x, y = batch(source, random_crop=False)
                _, loss = model(x, y)
                losses.append(float(loss))
            result[name] = sum(losses) / len(losses)
        model.train()
        return result

    best_validation_loss = math.inf
    best_step = None
    stale_evaluations = 0
    candidate_path = args.output_dir / f".checkpoint-{os.getpid()}.candidate.pt"
    for step in range(1, args.steps + 1):
        warmup = min(args.warmup_steps, args.steps)
        if warmup and step <= warmup:
            lr_scale = step / warmup
        else:
            progress = (step - warmup) / max(1, args.steps - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            lr_scale = args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * lr_scale
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.gradient_accumulation_steps):
            x, y = batch(train_pool)
            _, loss = model(x, y)
            (loss / args.gradient_accumulation_steps).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            losses = estimate_loss()
            print(
                f"step={step} train_loss={losses['train']:.4f} "
                f"validation_loss={losses['validation']:.4f} "
                f"validation_perplexity={math.exp(min(losses['validation'], 20)):.2f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )
            improved = best_validation_loss - losses["validation"] > args.min_delta
            if improved:
                best_validation_loss = losses["validation"]
                stale_evaluations = 0
                temporary_checkpoint = candidate_path.with_suffix(".pt.tmp")
                torch.save(
                    {"model_config": asdict(config),
                     "model_state_dict": model.state_dict(),
                     "optimizer_state_dict": optimizer.state_dict(),
                     "step": step, "seed": args.seed,
                     "validation_loss": best_validation_loss,
                     "provenance": "random initialization; no pretrained weights"},
                    temporary_checkpoint,
                )
                os.replace(temporary_checkpoint, candidate_path)
                best_step = step
            else:
                stale_evaluations += 1
            if (
                args.early_stopping_patience
                and stale_evaluations >= args.early_stopping_patience
            ):
                print(
                    f"early_stop step={step} best_validation_loss={best_validation_loss:.4f} "
                    f"patience={args.early_stopping_patience}",
                    flush=True,
                )
                break

    if not candidate_path.is_file():
        raise RuntimeError("Training ended without producing a validation-selected checkpoint.")
    os.replace(candidate_path, args.output_dir / "checkpoint.pt")
    run_manifest.update({
        "status": "completed",
        "actual_steps": step,
        "best_step": best_step,
        "best_validation_loss": best_validation_loss,
    })
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(
        f"published_checkpoint={args.output_dir / 'checkpoint.pt'} "
        f"best_step={best_step} best_validation_loss={best_validation_loss:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

