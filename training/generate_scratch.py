"""Generate a continuation with a saved Novyrix scratch-LM checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from scratch_lm import ByteTokenizer, ModelConfig, NovyrixScratchLM, choose_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = ModelConfig(**payload["model_config"])
    model = NovyrixScratchLM(config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    tokenizer = ByteTokenizer()
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    input_ids = torch.tensor(
        [[config.bos_id, *prompt_ids]], dtype=torch.long, device=device
    )
    generated = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
