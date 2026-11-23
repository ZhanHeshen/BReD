"""Train a small character-level nanoGPT on Tiny Shakespeare."""
import argparse
import math
from pathlib import Path

import torch

from bred import BReDAdamW, BReDMuon, BReDNAdamW, BReDMARS, BReDSOAP
from model import GPT, GPTConfig


MUON_WARMUP_STEPS = 100
MUON_DECAY_STEPS = 2000


def muon_lr_multiplier(completed_steps):
    if completed_steps < MUON_WARMUP_STEPS:
        return (completed_steps + 1) / (MUON_WARMUP_STEPS + 1)
    decay_step = min(completed_steps, MUON_DECAY_STEPS) - MUON_WARMUP_STEPS
    progress = decay_step / (MUON_DECAY_STEPS - MUON_WARMUP_STEPS)
    return 0.01 + 0.99 * (1 + math.cos(math.pi * progress)) / 2


def make_optimizers(model, name, state_format="mxfp4"):
    # The output head shares its weight with the token embedding.
    embeddings = [model.transformer.wte.weight, model.transformer.wpe.weight]
    embedding_ids = {id(p) for p in embeddings}
    matrices = [p for p in model.parameters() if p.ndim == 2 and id(p) not in embedding_ids]
    embedding_group = {"params": embeddings, "weight_decay": 0.1, "quantization": False}
    vector_group = {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0}
    lr, betas = 3e-4, (0.9, 0.99)
    if name in ("muon", "bred-muon"):
        return {
            "optimizer": BReDMuon(matrices, lr=3e-3, momentum=0.95, weight_decay=0.1,
                                  quantization=(name == "bred-muon"), state_format=state_format),
            "aux_optimizer": torch.optim.AdamW([embedding_group, vector_group], lr=lr, betas=betas),
        }
    groups = [embedding_group, {"params": matrices, "weight_decay": 0.1}, vector_group]
    if name in ("nadamw", "bred-nadamw"):
        return {"optimizer": BReDNAdamW(groups, lr=lr, betas=betas,
                                        quantization=(name == "bred-nadamw"), state_format=state_format)}
    if name in ("mars", "bred-mars"):
        return {"optimizer": BReDMARS(groups, lr=lr, betas=(0.95, 0.99), gamma=0.025,
                                     lr_1d=lr, betas_1d=betas, weight_decay_1d=0.0,
                                     quantization=(name == "bred-mars"), state_format=state_format)}
    if name in ("soap", "bred-soap"):
        return {"optimizer": BReDSOAP(groups, lr=lr, betas=(0.95, 0.95), shampoo_beta=0.95,
                                     precondition_frequency=10, quantization=(name == "bred-soap"), state_format=state_format)}
    optimizer_cls = BReDAdamW if name == "bred-adam" else torch.optim.AdamW
    kwargs = {"state_format": state_format} if name == "bred-adam" else {}
    return {"optimizer": optimizer_cls(groups, lr=lr, betas=betas, **kwargs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("input.txt"))
    parser.add_argument("--optimizer", choices=("adamw", "bred-adam", "muon", "bred-muon",
                                                "nadamw", "bred-nadamw", "mars", "bred-mars",
                                                "soap", "bred-soap"), default="bred-adam")
    parser.add_argument("--state-format", choices=("mxfp4", "mxint8"), default="mxfp4",
                        help="BReD optimizer-state format")
    parser.add_argument("--seed", type=int, default=42, help="model initialization and training batches")
    parser.add_argument("--steps", type=int, help="total updates, including resumed steps (AdamW: 1000, others: 2000)")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.state_format != "mxfp4" and not args.optimizer.startswith("bred-"):
        parser.error("--state-format applies only to BReD optimizers")
    use_muon = args.optimizer in ("muon", "bred-muon")
    if args.steps is None:
        args.steps = 1000 if args.optimizer in ("adamw", "bred-adam") else 2000
    if args.steps < 1 or (args.resume and not args.checkpoint):
        parser.error("--steps must be positive and --resume needs --checkpoint")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.optimizer in ("bred-adam", "bred-muon", "bred-nadamw", "bred-mars", "bred-soap") and device != "cuda":
        parser.error("BReD requires a CUDA GPU and Triton")

    text = args.data.read_text(encoding="utf-8")
    vocab = {c: i for i, c in enumerate(sorted(set(text)))}
    tokens = torch.tensor([vocab[c] for c in text])
    train, val = tokens.tensor_split([int(len(tokens) * 0.9)])
    block_size, batch_size = 128, 16
    if min(len(train), len(val)) <= block_size:
        parser.error("both data splits must be longer than 128 characters")
    offsets = torch.arange(block_size + 1)

    def batch(split, rng):
        start = torch.randint(len(split) - block_size, (batch_size, 1), generator=rng)
        window = split[start + offsets].to(device)
        return window[:, :-1], window[:, 1:].contiguous()

    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    model = GPT(GPTConfig(vocab_size=len(vocab), block_size=block_size, n_layer=4,
                          n_head=4, n_embd=128, dropout=0.0)).to(device)
    optimizers = make_optimizers(model, args.optimizer, args.state_format)
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if ckpt["optimizer_name"] != args.optimizer or ckpt["vocab"] != vocab:
            parser.error("checkpoint optimizer or vocabulary differs from this run")
        if args.optimizer.startswith("bred-"):
            formats = {group.get("state_format", "mxfp4") for group in ckpt["optimizer"]["param_groups"]}
            if formats != {args.state_format}:
                parser.error("checkpoint state format differs from --state-format")
        model.load_state_dict(ckpt["model"])
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(ckpt[name])
        rng.set_state(ckpt["rng"])
        start_step = ckpt["step"]
        del ckpt

    for step in range(start_step + 1, args.steps + 1):
        if use_muon:
            multiplier = muon_lr_multiplier(step - 1)
            for optimizer in optimizers.values():
                for group in optimizer.param_groups:
                    group["lr"] = optimizer.defaults["lr"] * multiplier
        model.zero_grad(set_to_none=True)
        _, loss = model(*batch(train, rng))
        loss.backward()
        for optimizer in optimizers.values():
            optimizer.step()
        if step % 100 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                eval_rng = torch.Generator().manual_seed(0)
                val_loss = torch.stack([model(*batch(val, eval_rng))[1] for _ in range(8)]).mean()
            model.train()
            state_bytes = sum(tensor_bytes(opt.state) for opt in optimizers.values())
            print(f"step {step}  train_loss {loss.item():.4f}  val_loss {val_loss.item():.4f}"
                  f"  optimizer_state_bytes {state_bytes}", flush=True)
            if args.checkpoint:
                temporary = args.checkpoint.with_suffix(".tmp")
                torch.save(dict(step=step, model=model.state_dict(), optimizer_name=args.optimizer,
                                rng=rng.get_state(), vocab=vocab,
                                **{k: opt.state_dict() for k, opt in optimizers.items()}), temporary)
                temporary.replace(args.checkpoint)


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(item) for item in value)
    return 0


if __name__ == "__main__":
    main()
