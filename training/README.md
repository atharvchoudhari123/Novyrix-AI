# Novyrix training paths

Novyrix currently has two distinct training paths. Do not confuse their provenance.

## Existing fine-tuning path

The existing tier scripts use a pretrained base checkpoint and fine-tune it on Novyrix examples. The model weights are therefore derived from that base model; this is not from-scratch pretraining.

Activate the environment and run the existing project command:

```bash
./train_all.sh
```

## Experimental from-scratch path

`training/scratch_lm.py` is an initial, separate decoder-only Transformer trainer. It initializes model weights randomly and uses a documented UTF-8 byte codec; it does not load Transformers or any pretrained checkpoint.

You must supply text files you have permission to use. The current trainer is an experimental foundation and is not yet a capable assistant.

```bash
python3 training/scratch_lm.py \
  --corpus path/to/approved-corpus.txt \
  --output-dir training/output/novyrix-scratch \
  --steps 10000
```

By default it selects CUDA, then Apple MPS, then CPU when available. Training from scratch can take substantial time. Small corpora and small models are useful for pipeline smoke checks only. The script requires enough encoded tokens for both training and validation splits; it does not download a corpus.

The output directory contains a model checkpoint, model configuration, tokenizer description, and training manifest. The current model has not yet been smoke-trained, integrated into the chat runtime, or evaluated as an assistant. Do not point the production chat runtime at an unvalidated checkpoint.

Before scaling up, add corpus license/provenance review, deduplication, stable train/validation splits, checkpoint resume, and robust evaluation. See [the model design and stage gates](own-model-design.md).
