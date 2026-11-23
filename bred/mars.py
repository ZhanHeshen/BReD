# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# Adapted from AGI-Arena/MARS, commit 4831e28eba863a4e69d7e9474ff8e73bc1b410fe,
# MARS/optimizers/mars.py. Limited to approximate MARS-AdamW with AdamW for vectors;
# stores configuration in parameter groups and adds optional BReD moment storage.
import math

import torch

from ._moments import PackedMoments


class _MARSAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay, gamma,
                 lr_1d, betas_1d, weight_decay_1d):
        if lr <= 0 or eps < 0 or weight_decay < 0 or gamma < 0:
            raise ValueError("lr must be positive; eps, weight_decay and gamma must be nonnegative")
        if lr_1d is None:
            lr_1d = lr
        if lr_1d < 0 or weight_decay_1d < 0:
            raise ValueError("lr_1d and weight_decay_1d must be nonnegative")
        for pair in (betas, betas_1d):
            if len(pair) != 2 or not all(0 <= b < 1 for b in pair):
                raise ValueError("betas and betas_1d must contain two values in [0, 1)")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                      gamma=gamma, lr_1d_factor=lr_1d / lr,
                                      betas_1d=betas_1d, weight_decay_1d=weight_decay_1d))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["last_grad"] = torch.zeros_like(p)
                state["step"] += 1
                step = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                lr, wd = group["lr"], group["weight_decay"]
                if p.ndim == 2:
                    beta1, beta2 = group["betas"]
                    corrected = (grad - state["last_grad"]).mul(
                        group["gamma"] * (beta1 / (1 - beta1))).add(grad)
                    norm = corrected.norm()
                    if norm > 1:
                        corrected = corrected / norm
                else:
                    beta1, beta2 = group["betas_1d"]
                    lr *= group["lr_1d_factor"]
                    wd = group["weight_decay_1d"]
                    corrected = grad
                m.mul_(beta1).add_(corrected, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(corrected, corrected, value=1 - beta2)
                denom = v.sqrt().mul(1 / math.sqrt(1 - beta2 ** step)).add(
                    group["eps"]).mul(1 - beta1 ** step)
                p.add_(-lr * p.mul(wd).add(m.div(denom)))
                state["last_grad"] = grad.clone()


class BReDMARS(PackedMoments, _MARSAdamW):
    """Approximate MARS-AdamW with MXFP4 or MXINT8 moments and an FP32 previous gradient.

    Matrices use MARS; other parameters use AdamW with the *_1d settings.
    Requires float32 parameters. Set quantization=False globally or per group
    to keep dense moments. Exact MARS, AMSGrad, Lion and Shampoo are not included.
    """

    def __init__(self, params, lr=3e-3, betas=(0.95, 0.99), eps=1e-8,
                 weight_decay=0.0, gamma=0.025, lr_1d=None,
                 betas_1d=(0.9, 0.95), weight_decay_1d=0.1, *,
                 quantization=True, global_seed=0, state_format="mxfp4"):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                         gamma=gamma, lr_1d=lr_1d, betas_1d=betas_1d,
                         weight_decay_1d=weight_decay_1d,
                         quantization=quantization, global_seed=global_seed,
                         state_format=state_format)
