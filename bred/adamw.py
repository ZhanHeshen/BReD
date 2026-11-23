# Adam / AdamW: https://arxiv.org/abs/1412.6980 and https://arxiv.org/abs/1711.05101
# Reference implementation: PyTorch AdamW (see THIRD_PARTY_LICENSES).
# The local update is checked against torch.optim.AdamW in tests/test_adamw.py.

import math

import torch

from ._optimizer import PackedStateOptimizer, state_elem_format


class BReDAdamW(PackedStateOptimizer):
    """AdamW with BReD storage for its two moments. Set quantization=False
    on a parameter group to keep its moments in full precision.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, amsgrad=False, *, global_seed=0, state_format="mxfp4"):
        if lr < 0 or eps < 0 or weight_decay < 0:
            raise ValueError("lr, eps and weight_decay must be nonnegative")
        if not all(0 <= beta < 1 for beta in betas) or len(betas) != 2:
            raise ValueError("betas must contain two values in [0, 1)")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay, amsgrad=amsgrad,
                                      global_seed=global_seed, quantization=True,
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
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                tensor_id += 1  # Stable even when a parameter has no gradient.
                grad = param.grad
                if grad is None or param.numel() == 0:
                    continue
                if grad.is_sparse or param.is_complex():
                    raise RuntimeError("BReDAdamW requires dense, real gradients")
                packed = group["quantization"] and param.ndim > 0
                if packed:
                    if not param.is_cuda:
                        raise ValueError("BReD packed state requires a CUDA tensor")
                    from .quantization import block_dither, dequantize_mx_tri, quantize_mx_tri

                    n_blocks = math.prod(param.shape[:-1]) * ((param.shape[-1] + 31) // 32)

                state = self.state[param]
                step = state.get("step", 0)
                names = ("exp_avg", "exp_avg_sq")
                moments = []
                for state_id, name in enumerate(names):
                    if not step:
                        moment = torch.zeros_like(param, dtype=torch.float32)
                    elif packed:
                        noise = block_dither(group["global_seed"], n_blocks, tensor_id,
                                             state_id, step, param.device)
                        moment = dequantize_mx_tri(
                            state[name + "_packed"], state[name + "_shared_exp"],
                            elem_format=elem_format, round="subtract_dither", dither_rng=noise, output_shape=param.shape)
                    else:
                        moment = state[name]
                    moments.append(moment)

                m, v = moments
                if packed and step:
                    v.abs_()
                grad = grad.float()
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                step += 1
                variance = v
                if group["amsgrad"]:
                    if step == 1:
                        state["max_exp_avg_sq"] = torch.zeros_like(v)
                    variance = state["max_exp_avg_sq"]
                    torch.maximum(variance, v, out=variance)
                denom = variance.sqrt().div_(math.sqrt(1 - beta2 ** step)).add_(group["eps"])
                lr = group["lr"]
                param.mul_(1 - lr * group["weight_decay"])
                param.addcdiv_(m, denom, value=-lr / (1 - beta1 ** step))

                for state_id, (name, moment) in enumerate(zip(names, moments)):
                    if packed:
                        noise = block_dither(group["global_seed"], n_blocks, tensor_id,
                                             state_id, step, param.device)
                        state[name + "_packed"], state[name + "_shared_exp"] = quantize_mx_tri(
                            moment, elem_format=elem_format, round="subtract_dither", dither_rng=noise)
                    else:
                        state[name] = moment
                state["step"] = step
        return loss
