"""Packed moment storage around a dense optimizer's update."""
import math

import torch

from ._optimizer import state_elem_format


class PackedMoments:
    def __init__(self, params, *, quantization=True, global_seed=0, state_format="mxfp4", **kwargs):
        super().__init__(params, **kwargs)
        self.defaults.update(quantization=quantization, global_seed=global_seed, state_format=state_format)
        for group in self.param_groups:
            group.setdefault("quantization", quantization)
            group.setdefault("global_seed", global_seed)
            state_elem_format(group.setdefault("state_format", state_format))

    def load_state_dict(self, state_dict):
        for group in state_dict["param_groups"]:
            state_elem_format(group.get("state_format", "mxfp4"))
        super().load_state_dict(state_dict)
        # Keep codes/exponents int8. Native NAdam's non-capturable scalars live on CPU.
        for group, saved in zip(self.param_groups, state_dict["param_groups"]):
            for param, index in zip(group["params"], saved["params"]):
                for key, value in state_dict["state"].get(index, {}).items():
                    if torch.is_tensor(value):
                        device = "cpu" if key in ("step", "mu_product") else param.device
                        self.state[param][key] = value.to(device=device, copy=True)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        packed = []
        tensor_id = -1
        for group in self.param_groups:
            elem_format = state_elem_format(group.get("state_format", "mxfp4"))
            for p in group["params"]:
                tensor_id += 1  # Includes excluded parameters and missing gradients.
                if p.grad is None:
                    continue
                if p.dtype != torch.float32 or p.grad.is_sparse:
                    raise ValueError("this optimizer requires float32 parameters and dense gradients")
                if group["quantization"] and p.ndim > 0 and p.numel():
                    if not p.is_cuda:
                        raise ValueError("BReD packed state requires a CUDA tensor")
                    blocks = math.prod(p.shape[:-1]) * ((p.shape[-1] + 31) // 32)
                    packed.append((p, tensor_id, blocks, group["global_seed"], elem_format))

        if packed:
            from .quantization import block_dither, dequantize_mx_tri, quantize_mx_tri

        for p, tensor_id, blocks, seed, elem_format in packed:
            state = self.state[p]
            if "moment_step" not in state:
                continue
            for state_id, name in enumerate(("exp_avg", "exp_avg_sq")):
                noise = block_dither(seed, blocks, tensor_id, state_id, state["moment_step"], p.device)
                value = dequantize_mx_tri(
                    state[name + "_packed"], state[name + "_shared_exp"],
                    elem_format=elem_format, round="subtract_dither", dither_rng=noise, output_shape=p.shape)
                state[name] = value.abs_() if state_id == 1 else value

        result = super().step()

        for p, tensor_id, blocks, seed, elem_format in packed:
            state = self.state[p]
            step = state.get("moment_step", 0) + 1
            for state_id, name in enumerate(("exp_avg", "exp_avg_sq")):
                noise = block_dither(seed, blocks, tensor_id, state_id, step, p.device)
                state[name + "_packed"], state[name + "_shared_exp"] = quantize_mx_tri(
                    state[name], elem_format=elem_format, round="subtract_dither", dither_rng=noise)
                del state[name]
            state["moment_step"] = step
        return loss if closure is not None else result
