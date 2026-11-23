# Copyright (c) 2024 Keller Jordan
# Newton-Schulz and momentum updates adapted from KellerJordan/Muon (MIT).
# Source: https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py
# See THIRD_PARTY_LICENSES.
# Changes: FP32 arithmetic, AdamW-matched scaling, and BReD momentum storage.

import math

import torch

from ._optimizer import PackedStateOptimizer, state_elem_format


def _orthogonalize(update):
    """Five quintic Newton–Schulz steps on a Frobenius-normalized matrix."""
    transposed = update.size(0) > update.size(1)
    x = update.T if transposed else update
    x = x / (x.norm() + 1e-7)
    for _ in range(5):
        gram = x @ x.T
        polynomial = -4.775 * gram + 2.0315 * (gram @ gram)
        x = 3.4445 * x + polynomial @ x
    return x.T if transposed else x


class BReDMuon(PackedStateOptimizer):
    """Muon for hidden weight matrices, with optional MXFP4 or MXINT8 momentum storage.

    Use AdamW separately for embeddings, the output head, and vector parameters.
    The matrix update uses Nesterov momentum and AdamW-matched RMS scaling.
    """

    def __init__(self, params, lr=3e-4, momentum=0.95, weight_decay=0.1, *,
                 quantization=True, global_seed=0, state_format="mxfp4"):
        if lr < 0 or weight_decay < 0 or not 0 <= momentum < 1:
            raise ValueError("expected nonnegative lr and weight_decay, and momentum in [0, 1)")
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                                      quantization=quantization, global_seed=global_seed,
                                      state_format=state_format))
        if any(p.ndim != 2 for group in self.param_groups for p in group["params"]):
            raise ValueError("Muon expects 2D hidden weight matrices")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        tensor_id = -1
        for group in self.param_groups:
            elem_format = state_elem_format(group.get("state_format", "mxfp4"))
            beta, lr = group["momentum"], group["lr"]
            for p in group["params"]:
                tensor_id += 1
                if p.grad is None or p.numel() == 0:
                    continue
                if p.ndim != 2 or p.is_complex() or p.grad.is_sparse:
                    raise ValueError("Muon expects real matrices with dense gradients")
                packed = group["quantization"]
                if packed:
                    if not p.is_cuda:
                        raise ValueError("BReD packed state requires a CUDA tensor")
                    from . import quantization as fp4

                    n_blocks = p.size(0) * ((p.size(1) + 31) // 32)

                state = self.state[p]
                step = state.get("step", 0)
                if not step:
                    buf = torch.zeros_like(p, dtype=torch.float32)
                elif packed:
                    noise = fp4.block_dither(group["global_seed"], n_blocks, tensor_id, 0, step, p.device)
                    buf = fp4.dequantize_mx_tri(
                        state["momentum_packed"], state["momentum_shared_exp"],
                        elem_format=elem_format, round="subtract_dither", dither_rng=noise, output_shape=p.shape)
                else:
                    buf = state["momentum_buffer"]

                grad = p.grad.float()
                buf.lerp_(grad, 1 - beta)
                update = _orthogonalize(grad.lerp(buf, beta))
                scale = 0.2 * math.sqrt(max(p.shape))
                p.mul_(1 - lr * group["weight_decay"]).add_(update, alpha=-lr * scale)

                step += 1
                if packed:
                    noise = fp4.block_dither(group["global_seed"], n_blocks, tensor_id, 0, step, p.device)
                    state["momentum_packed"], state["momentum_shared_exp"] = fp4.quantize_mx_tri(
                        buf, elem_format=elem_format, round="subtract_dither", dither_rng=noise)
                else:
                    state["momentum_buffer"] = buf
                state["step"] = step
        return loss
