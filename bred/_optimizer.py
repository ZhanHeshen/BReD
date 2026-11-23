import torch


def state_elem_format(state_format):
    if state_format == "mxfp4":
        return "fp4_e2m1"
    if state_format == "mxint8":
        return "int8"
    raise ValueError("state_format must be mxfp4 or mxint8")


class PackedStateOptimizer(torch.optim.Optimizer):
    def __init__(self, params, defaults):
        super().__init__(params, defaults)
        for group in self.param_groups:
            state_elem_format(group.get("state_format", "mxfp4"))

    def load_state_dict(self, state_dict):
        for group in state_dict["param_groups"]:
            state_elem_format(group.get("state_format", "mxfp4"))
        # PyTorch normally casts optimizer tensors to the parameter dtype.
        # Restore our tensors directly so packed codes/exponents stay int8
        # and dense moments stay float32, including for bf16 parameters.
        metadata = {i: {k: v for k, v in state.items() if not torch.is_tensor(v)}
                    for i, state in state_dict["state"].items()}
        super().load_state_dict({**state_dict, "state": metadata})

        def restore(value, device):
            if torch.is_tensor(value):
                return value.to(device=device, copy=True)
            if isinstance(value, dict):
                return {key: restore(item, device) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(restore(item, device) for item in value)
            return value

        for group, saved in zip(self.param_groups, state_dict["param_groups"]):
            for param, index in zip(group["params"], saved["params"]):
                for key, value in state_dict["state"].get(index, {}).items():
                    self.state[param][key] = restore(value, param.device)
