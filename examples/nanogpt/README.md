# nanoGPT on Tiny Shakespeare

Train a small GPT model and compare FP32 optimizer states with BReD's low-precision storage.
BReD requires a CUDA GPU and Triton.

## Installation

Run from the repository root. If you already have an environment with CUDA-enabled
PyTorch and Triton, activate it and run only `python -m pip install -e .`.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
```

## Prepare data

Keep the environment active and run the remaining commands from `examples/nanogpt`:

```sh
cd examples/nanogpt
test -s input.txt || curl -fL https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o input.txt
```

## Training

Start with a short run:

```sh
python train.py --optimizer bred-adam --steps 10
```

Then compare an optimizer with and without BReD, using the same seed:

```sh
python train.py --optimizer adamw --seed 0
python train.py --optimizer bred-adam --seed 0
```

Choose another pair with `--optimizer`:

| Optimizer | FP32 | BReD |
|---|---|---|
| AdamW | `adamw` | `bred-adam` |
| Muon | `muon` | `bred-muon` |
| NAdamW | `nadamw` | `bred-nadamw` |
| MARS | `mars` | `bred-mars` |
| SOAP | `soap` | `bred-soap` |

Use `--steps` to set the training length (default: 1,000 for AdamW, 2,000 for
others), `--seed` to change the random seed (default: 42), and `--data` to use
another text file. Training and validation loss and optimizer-state bytes are
printed every 100 steps and at the final step.

## Save and resume

Checkpoints are saved only when `--checkpoint` is provided:

```sh
python train.py --optimizer bred-adam --steps 100 --checkpoint bred.pt
python train.py --optimizer bred-adam --steps 1000 --checkpoint bred.pt --resume
```

On resume, `--steps` is the total target step count. Use the same optimizer, state format and data.

The model is adapted from [nanoGPT](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/model.py)
under the [MIT license](../../THIRD_PARTY_LICENSES).
