# Novyrix from-scratch model: design and staged implementation

## Goal

Build a language model whose neural-network weights are initialized and trained by the Novyrix project. Do not load pretrained Qwen, Llama, GPT, or other foundation-model weights in the scratch path.

This is distinct from the existing product/agent stack. The current Novyrix tiers use a pretrained Qwen checkpoint by default and the current standard training pipeline fine-tunes pretrained checkpoints. Keep that path available as a clearly labeled baseline while the scratch model is developed on this branch.

## What “our own model” means here

- Novyrix owns the model definition, initialization, training recipe, and resulting checkpoints.
- Scratch language-model weights start from random initialization; no pretrained language-model weights are imported.
- Data sources, licenses, filtering, and deduplication must be recorded before any substantial pretraining run.
- Evaluation reports the exact checkpoint, data, compute, and metrics. Do not call an untrained/random checkpoint a capable assistant.
- Third-party libraries may provide tensor operations and training utilities. Using PyTorch does not make pretrained weights of another model part of Novyrix.

## Important capability trade-off

A small model pretrained from scratch on a small corpus will initially perform far worse than the existing Qwen-based baseline. The initial implementation proves the architecture and training path, not useful general-purpose or coding capability. Meaningful language capability requires large, clean, legally usable data and substantial compute. Preserve the baseline until the scratch model passes evaluation.

## Current experimental implementation

`training/scratch_lm.py` contains an initial decoder-only Transformer and next-token training loop. It initializes its own weights and does not use Transformers or load a pretrained checkpoint. Its tokenizer is a transparent UTF-8 byte codec: IDs 3–258 represent bytes, with separate BOS/EOS/PAD IDs. This avoids hidden dependencies on a third-party tokenizer but is less compact than BPE and is not claimed to be a trained tokenizer.

Example command (use only text files you have permission to train on):

```bash
python3 training/scratch_lm.py \
  --corpus path/to/approved-corpus.txt \
  --output-dir training/output/novyrix-scratch \
  --steps 10000
```

This code has been added but has not yet been smoke-trained or independently validated. The generated checkpoint must not be used as a capable chat model until the training path and evaluations are complete.

## Target system

1. **Tokenizer:** Stage 1 uses the documented byte codec. A future learned BPE tokenizer can be trained from an approved corpus; its vocabulary, merges, special tokens, corpus manifest, and checksum must be stored with every checkpoint.
2. **Architecture:** Implement and validate a compact decoder-only causal Transformer with learned positions, causal self-attention, MLP blocks, normalization, and documented tied embeddings. Initialize all weights randomly and keep configuration versioned.
3. **Pretraining:** Train next-token prediction on deduplicated, licensed text/code corpora. Add deterministic data manifests, packed sequences, validation splits, checkpoint/resume, gradient accumulation, mixed precision where supported, and loss/throughput reporting before scaling up.
4. **Instruction tuning:** Only after healthy pretraining, tune on a separately licensed and documented instruction dataset. Keep pretraining and instruction-tuning checkpoints distinct.
5. **Inference:** Add a separate runtime/registry entry for the scratch checkpoint and tokenizer. Never silently fall back to a Qwen checkpoint under a Novyrix scratch-model name.
6. **Evaluation:** Track validation loss/perplexity and task-level evaluations, including code generation and factuality. Report failures and compare with the existing Qwen baseline at appropriate resource budgets.
7. **Product integration:** Keep the current Novyrix UI, API, agent, and tools independent of model choice. Expose actual model provenance and capability status in configuration/UI.

## Stages and gates

- **Stage 0 — provenance:** Document all current base models and label the existing Qwen-backed path as fine-tuned. Done in this branch's README.
- **Stage 1 — minimal scratch LM:** Initial byte codec, randomly initialized Transformer, and trainer code are present. Still required: syntax/smoke training, checkpoint save/load verification, deterministic generation check, and basic reproducible evaluation. This only validates the pipeline; it does not create a capable assistant.
- **Stage 2 — data and pretraining:** Proceed only with a reviewed, licensed corpus and a compute/time estimate. Record corpus hashes and training configuration.
- **Stage 3 — instruction tuning and evaluation:** Add instruction tuning only after pretraining is healthy; evaluate coding, factuality, and chat separately.
- **Stage 4 — optional rollout:** Make the scratch model selectable only after quality and resource requirements are documented. Keep a visible baseline option and never represent fine-tuned third-party weights as scratch-trained.

## Acceptance criteria for a real scratch checkpoint

- The model architecture is instantiated from a Novyrix config with random initialization.
- No third-party pretrained checkpoint is fetched or loaded in the scratch training or inference path.
- The tokenizer format is documented and saved with the checkpoint (the Stage 1 byte codec is not described as a trained BPE tokenizer).
- Training logs and manifests identify data provenance, configuration, seed, and steps.
- A clean environment can load the resulting model and tokenizer using only saved artifacts.
- Evaluation is reported honestly, with no capability claim based only on model branding or parameter count.

## Current status

The initial architecture/trainer code is committed on this branch but has not yet been smoke-trained or independently validated. There is no verified scratch-trained Novyrix checkpoint, substantial licensed pretraining corpus, or evidence of assistant-level capability yet.
