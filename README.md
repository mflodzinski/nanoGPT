# nanoGPT

A small PyTorch implementation of a GPT-style language model, inspired by Andrej Karpathy's nanoGPT and educational transformer walkthroughs.

The repo contains a compact transformer model, a simple tokenizer/data loader, a YAML config, and a notebook for training on `data/input.txt` and generating text.

## Structure

- `nano-gpt.ipynb` - notebook entry point for training and generation
- `src/transformer.py` - transformer blocks, attention, training loop, and generation
- `src/data.py` - dataset loading, tokenization, train/validation split, and batching
- `config/config.yaml` - model, data, training, and generation settings
- `data/input.txt` - training text

## Run

Install the main dependencies:

```bash
pip install torch pyyaml jupyter
```

Then open the notebook:

```bash
jupyter notebook nano-gpt.ipynb
```

Adjust hyperparameters in `config/config.yaml`, run the notebook cells, and inspect the generated sample text at the end.

## Credit

The idea and learning path come from Andrej Karpathy's work on small GPT implementations, especially nanoGPT.
