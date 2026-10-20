# Novyrix scratch training

Novyrix's text models use a custom decoder-only Transformer initialized from random weights. The training and inference paths do not load any pretrained language-model checkpoint.

## Train the three Novyrix tiers

From the project root, activate the virtual environment and run:

```bash
./train_all.sh
```

This generates small demonstration datasets and trains separate architectures:

- 3.2: width 256, 4 layers, 4 attention heads
- 4.0: width 384, 6 layers, 6 attention heads
- 5.7: width 512, 8 layers, 8 attention heads

The default is 1,000 training steps per tier. Change it with `NOVYRIX_TRAIN_STEPS`; `NOVYRIX_EVAL_EVERY` and `NOVYRIX_BATCH_SIZE` can also be set before starting. Training time depends on hardware and these settings. The generated sample data is only suitable for pipeline experiments, not for producing a reliable assistant. Use only larger data you have the right to train on before expecting useful capabilities.

The model uses a reversible UTF-8 byte tokenizer and saves a checkpoint, model configuration, tokenizer description, and training manifest in each `training/output/novyrix-*-scratch` folder. The API runtime loads only these custom scratch checkpoints and will report a clear missing-checkpoint error until training has produced them.

Generate a continuation manually:

```bash
python3 training/generate_scratch.py \
  --checkpoint training/output/novyrix-3.2-scratch/checkpoint.pt \
  --prompt "Hello"
```

## Limitations

A small randomly initialized model trained on a few hundred examples will produce poor, unreliable language. The architecture is Novyrix's own; capability still requires substantial, appropriately licensed pretraining data, compute, instruction tuning, and evaluation. The tier names do not imply benchmark performance.

