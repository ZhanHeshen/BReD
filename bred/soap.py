# Copyright (c) 2024 Nikhil Vyas
# SPDX-License-Identifier: MIT
# Adapted from nikhilvyas/SOAP, commit a1e553530fde97d0e6b307d7c82ac6d38b072340.
# Adds BReD storage; omits dimension merging and channels-last reshaping.
import math

import torch

from ._optimizer import PackedStateOptimizer, state_elem_format


def _project(value, bases, back=False):
    for basis in bases:
        if len(basis):
            value = torch.tensordot(value, basis, dims=([0], [1 if back else 0]))
        else:
            value = value.permute(*range(1, value.ndim), 0)
    return value


def _update_factors(grad, m, v, factors, bases, step, group):
    if bases is not None:
        m = _project(m, bases, back=True)
    beta = group["shampoo_beta"] if group["shampoo_beta"] >= 0 else group["betas"][1]
    for axis, factor in enumerate(factors):
        if len(factor):
            axes = [i for i in range(grad.ndim) if i != axis]
            outer = torch.tensordot(grad, grad, dims=(axes, axes))
            factor.lerp_(outer, 1 - beta)
    if bases is None:
        bases = []
        for factor in factors:
            if not len(factor):
                bases.append([])
                continue
            identity = torch.eye(factor.shape[0], device=factor.device)
            try:
                _, basis = torch.linalg.eigh(factor + 1e-30 * identity)
            except torch.linalg.LinAlgError:
                _, basis = torch.linalg.eigh(factor.double() + 1e-30 * identity)
                basis = basis.float()
            bases.append(basis.flip([1]))
    if step > 0 and step % group["precondition_frequency"] == 0:
        new_bases = []
        for axis, (factor, basis) in enumerate(zip(factors, bases)):
            if not len(factor):
                new_bases.append([])
                continue
            order = torch.argsort(torch.diag(basis.T @ factor @ basis), descending=True)
            v = v.index_select(axis, order)
            basis, _ = torch.linalg.qr(factor @ basis[:, order])
            new_bases.append(basis)
        bases = new_bases
    if step > 0:
        m = _project(m, bases)
    return m, v, bases


class BReDSOAP(PackedStateOptimizer):
    """SOAP with MXFP4 or MXINT8 first/second moments and preconditioner statistics.

    Eigenbases Q stay FP32. Reconstructed statistics are symmetrized. Requires
    float32 parameters; supports tensors without dimension merging. Set
    quantization=False globally or per parameter group to keep dense state.
    The first step initializes the eigenbases without updating parameters.
    """

    def __init__(self, params, lr=3e-3, betas=(0.95, 0.95), shampoo_beta=-1,
                 eps=1e-8, weight_decay=0.01, precondition_frequency=10,
                 max_precond_dim=10000, precondition_1d=False,
                 normalize_grads=False, correct_bias=True, *,
                 quantization=True, global_seed=0, state_format="mxfp4"):
        if lr < 0 or eps < 0 or weight_decay < 0:
            raise ValueError("lr, eps and weight_decay must be nonnegative")
        if len(betas) != 2 or not all(0 <= b < 1 for b in betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if shampoo_beta != -1 and not 0 <= shampoo_beta < 1:
            raise ValueError("shampoo_beta must be -1 or in [0, 1)")
        if (not isinstance(precondition_frequency, int) or precondition_frequency < 1
                or not isinstance(max_precond_dim, int) or max_precond_dim < 1):
            raise ValueError("precondition_frequency and max_precond_dim must be positive integers")
        super().__init__(params, dict(lr=lr, betas=betas, shampoo_beta=shampoo_beta,
                                      eps=eps, weight_decay=weight_decay,
                                      precondition_frequency=precondition_frequency,
                                      max_precond_dim=max_precond_dim, precondition_1d=precondition_1d,
                                      normalize_grads=normalize_grads, correct_bias=correct_bias,
                                      quantization=quantization, global_seed=global_seed,
                                      state_format=state_format))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        tensor_id = -1
        for group in self.param_groups:
            elem_format = state_elem_format(group.get("state_format", "mxfp4"))
            for p in group["params"]:
                tensor_id += 1
                if p.grad is None or not p.numel():
                    continue
                if p.dtype != torch.float32 or p.grad.is_sparse or p.ndim == 0:
                    raise ValueError("BReDSOAP requires non-scalar float32 tensors and dense gradients")
                packed = group["quantization"]
                if packed:
                    if not p.is_cuda:
                        raise ValueError("BReD packed state requires a CUDA tensor")
                    from .quantization import block_dither, quantize_mx_tri, dequantize_mx_tri

                    def noise(shape, state_id, storage_step):
                        blocks = math.prod(shape[:-1]) * ((shape[-1] + 31) // 32)
                        return block_dither(group["global_seed"], blocks, tensor_id,
                                             state_id, storage_step, p.device)

                    def unpack(data, state_id, storage_step):
                        codes, scales, shape = data
                        return dequantize_mx_tri(codes, scales, elem_format=elem_format, round="subtract_dither",
                                                 dither_rng=noise(shape, state_id, storage_step),
                                                 output_shape=shape)

                    def pack(value, state_id, storage_step):
                        codes, scales = quantize_mx_tri(value, elem_format=elem_format, round="subtract_dither",
                                                       dither_rng=noise(value.shape, state_id, storage_step))
                        return codes, scales, tuple(value.shape)

                state = self.state[p]
                step = state.get("step", 0)
                bases = state.get("Q")
                if step == 0:
                    m, v = torch.zeros_like(p), torch.zeros_like(p)
                elif packed:
                    m = unpack(state["exp_avg_packed"], 0, step + 1)
                    v = unpack(state["exp_avg_sq_packed"], 1, step + 1).abs_()
                else:
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                if bases is None:
                    factors = [torch.zeros(dim, dim, device=p.device)
                               if dim <= group["max_precond_dim"] and (p.ndim > 1 or group["precondition_1d"])
                               else [] for dim in p.shape]
                elif packed:
                    factors = []
                    for axis, data in enumerate(state["GG_packed"]):
                        value = unpack(data, axis + 2, step + 1) if data else None
                        factors.append((value + value.T) * 0.5 if value is not None else [])
                else:
                    factors = state["GG"]

                if bases is not None:
                    projected = _project(p.grad, bases)
                    beta1, beta2 = group["betas"]
                    step += 1
                    m.mul_(beta1).add_(projected, alpha=1 - beta1)
                    v.mul_(beta2).add_(projected.square(), alpha=1 - beta2)
                    denom = v.sqrt().add_(group["eps"])
                    step_size = group["lr"]
                    if group["correct_bias"]:
                        step_size = step_size * (1 - beta2 ** step) ** 0.5 / (1 - beta1 ** step)
                    update = _project(m / denom, bases, back=True)
                    if group["normalize_grads"]:
                        update = update / (1e-30 + torch.mean(update ** 2) ** 0.5)
                    p.add_(update, alpha=-step_size)
                    if group["weight_decay"] > 0:
                        p.add_(p, alpha=-group["lr"] * group["weight_decay"])

                m, v, bases = _update_factors(p.grad, m, v, factors, bases, step, group)
                state["step"], state["Q"] = step, bases
                if packed:
                    if step > 0:
                        state["exp_avg_packed"] = pack(m, 0, step + 1)
                        state["exp_avg_sq_packed"] = pack(v, 1, step + 1)
                    state["GG_packed"] = [pack(value, axis + 2, step + 1) if len(value) else []
                                          for axis, value in enumerate(factors)]
                else:
                    state["exp_avg"], state["exp_avg_sq"], state["GG"] = m, v, factors
        return loss
