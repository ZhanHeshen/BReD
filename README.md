# BReD
Code of paper "BReD: Block Replay Dithering for Stable Low-Bit EMA Optimizer States"

Block Replay Dithering (BReD) reduces optimizer-state memory by storing states
in packed MXFP4 by default and replaying the same dither when decoding them.

## Supported optimizers

| Optimizer | Class | Packed states |
|---|---|---|
| AdamW | [`BReDAdamW`](bred/adamw.py) | First and second moments |
| Muon | [`BReDMuon`](bred/muon.py) | Momentum |
| NAdamW | [`BReDNAdamW`](bred/nadamw.py) | First and second moments |
| MARS | [`BReDMARS`](bred/mars.py) | First and second moments; previous gradient stays FP32 |
| SOAP | [`BReDSOAP`](bred/soap.py) | First and second moments, preconditioner statistics (`L/R`); eigenbases `Q` stay FP32 |

NAdamW uses decoupled weight decay. MARS uses approximate MARS-AdamW for matrices
and AdamW for other parameters.

## Installation

With CUDA-enabled PyTorch and Triton installed, run from the repository root:

```sh
python -m pip install -e .
```

For a fresh environment, follow the [installation guide](examples/nanogpt/README.md#installation).

## Usage

Use a BReD optimizer in your training loop:

```python
from bred import BReDAdamW

optimizer = BReDAdamW(model.parameters(), lr=3e-4)
optimizer.zero_grad()
loss = loss_fn(model(inputs), targets)
loss.backward()
optimizer.step()
```

- Set `quantization=False` on a parameter group to keep its states in full precision.
- For Muon, pass only hidden 2D weight matrices. Use a separate AdamW for embeddings,
  the output head and vector parameters, as in the nanoGPT example.
- NAdamW, MARS and SOAP require float32 parameters.
- SOAP's first step initializes its eigenbases without updating parameters.
  Dimension merging is not included.

## nanoGPT example

The [nanoGPT guide](examples/nanogpt/README.md) provides setup and training commands
for all five optimizers on Tiny Shakespeare, including FP32 comparisons and
checkpoint resume.

## State storage

MXFP4 stores two elements per byte. Each block of 32 values shares one
exponent and one dither value. Packed codes and scales are retained between steps; FP32 reconstruction
buffers are temporary during updates.

Quantization and dither replay are implemented in
[quantization.py](bred/quantization.py).

To restore a checkpoint, recreate the same parameter groups in the same order
and call `optimizer.load_state_dict(...)`.

## Acknowledgements

Our optimizer implementations build on [PyTorch](https://github.com/pytorch/pytorch),
[Muon](https://github.com/KellerJordan/Muon), [MARS](https://github.com/AGI-Arena/MARS),
and [SOAP](https://github.com/nikhilvyas/SOAP). The example model is adapted from
[nanoGPT](https://github.com/karpathy/nanoGPT). We thank the authors for sharing their work.
Third-party notices are collected in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
