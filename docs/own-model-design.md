# Novyrix from-scratch model: design and staged implementation

## Goal

Build a language model whose tokenizer and neural-network weights are initialized and trained by the Novyrix project. Do not load a pretrained Qwen, Llama, GPT, or other foundation-model checkpoint as the source of the language-model weights.

This is distinct from the existing product/agent stack. The current Novyrix tiers use a pretrained Qwen checkpoint by default and the current training pipeline fine-tunes pretrained checkpoints. Keep that path available as a clearly labeled baseline while the new model is developed on this branch.

## What “our own model” means here

- Novyrix owns the model definition, tokenizer training artifacts, initialization, training recipe, and resulting checkpoints.
- The base language-model weights start from random initialization; no pretrained language-model weights are imported.
- Training-data sources, licenses, filtering, and deduplication are recorded.
- Evaluation reports the exact checkpoint, data, compute, and metrics. Do not call an untrained/random checkpoint an assistant or claim it is capable.
- Third-party libraries may implement standard tensor operations and training utilities. Using PyTorch does not make the pretrained weights of another model part of Novyrix.

## Important capability trade-off

A small model pretrained from scratch on a small corpus will initially perform far worse than the existing Qwen-based baseline. A scratch implementation proves ownership and creates a foundation for learning; it does not instantly provide a useful general-purpose or coding assistant. Meaningful language capability needs large, clean, legally usable data and substantial compute. Preserve the baseline until the new model has passed evaluation.

## Target system

1. **Tokenizer:** Train a new BPE tokenizer from the approved Novyrix training corpus. Save vocabulary, merges, special tokens, corpus manifest, and tokenizer checksum alongside each model checkpoint.
2. **Architecture:** Implement a compact decoder-only causal Transformer in PyTorch with learned/rotary positional handling, causal self-attention, MLP blocks, normalization, and tied or explicitly documented embeddings. Initialize all weights randomly. Keep the model config explicit and versioned.
3. **Pretraining:** Train next-token prediction on deduplicated, licensed text/code corpora. Use deterministic data manifests, packed sequences, validation splits, checkpoint/resume, gradient accumulation, mixed precision where supported, and loss/throughput reporting.
4. **Instruction tuning:** Only after pretraining, tune on a separately licensed and documented instruction dataset. Keep pretraining and instruction-tuning checkpoints distinct.
5. **Inference:** Add a separate runtime/registry entry for the scratch checkpoint and tokenizer. Never silently fall back to a Qwen checkpoint under a Novyrix scratch-model name.
6. **Evaluation:** Track validation loss/perplexity and task-level evaluations, including code generation and factuality. Include repeated-run settings and report failures. Compare with the existing Qwen baseline at appropriate resource budgets.
7. **Product integration:** Keep current Novyrix UI, API, agent, and tools independent of model choice. Expose the model's actual provenance and capability status in configuration/UI.

## Stages and gates

- **Stage 0 — provenance:** document all current base models and label the present Qwen-backed path as fine-tuned.
- **Stage 1 — minimal scratch LM:** tokenizer training, randomly initialized compact Transformer, synthetic/tiny-corpus smoke training, checkpoint save/load, and deterministic next-token inference. This validates the pipeline only; it is not a capable chat model.
- **Stage 2 — data and pretraining:** only proceed with a reviewed, licensed corpus and a compute/time estimate. Record corpus hashes and training configuration.
- **Stage 3 — instruction tuning and evaluation:** add instruction tuning only after pretraining is healthy; evaluate coding, factuality, and chat separately.
- **Stage 4 — optional rollout:** make the scratch model selectable only after quality and resource requirements are documented. Keep a visible baseline option and never represent fine-tuned third-party weights as scratch-trained.

## Acceptance criteria for a real scratch checkpoint

- The model architecture is instantiated from a Novyrix config with random initialization.
- No third-party pretrained checkpoint is fetched or loaded in the scratch train/inference path.
- The tokenizer is trained by the Novyrix pipeline and saved with the checkpoint.
- Training logs and manifests identify data provenance, configuration, seed, and steps.
- A clean environment can load the resulting model and tokenizer using only the saved artifacts.
- Evaluation is reported honestly, with no capability claim based only on model branding or parameter count.

## Current status

This document records the intended design. It does not claim that the scratch architecture, tokenizer, pretraining corpus, or trained checkpoint already exists. Implement and verify each stage on this branch before describing it as complete.
