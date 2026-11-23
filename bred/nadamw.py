# Wraps torch.optim.NAdam with decoupled weight decay; its update is not vendored.
# PyTorch dependency attribution: see THIRD_PARTY_LICENSES and README.md.

import torch

from ._moments import PackedMoments


class BReDNAdamW(PackedMoments, torch.optim.NAdam):
    """PyTorch NAdam with decoupled weight decay and MXFP4 or MXINT8 moment storage.

    Requires float32 parameters. Set quantization=False globally or per group
    to keep dense moments. Scalar Nesterov state stays in full precision.
    """

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, momentum_decay=0.004, *,
                 quantization=True, global_seed=0, state_format="mxfp4"):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                         momentum_decay=momentum_decay, decoupled_weight_decay=True,
                         foreach=False, quantization=quantization, global_seed=global_seed,
                         state_format=state_format)
