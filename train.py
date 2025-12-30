from __future__ import annotations

import os
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt_small import GPTConfig, GPTSmall


@dataclass
class TrainConfig:
    seed: int = 123
    device: str = "cpu"  # cpu|cuda|mps

    # data
    train_blocks_path: str = ""
    val_blocks_path: str = ""
    block_size: int = 64
    vocab_size: int = 0

    # optimization
    batch_size: int = 64
    max_steps: int = 10_000
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_steps: int = 200
    min_lr: float = 1e-5

    # eval/checkpoint
    eval_every: int = 500
    ckpt_every: int = 1000
    out_dir: str = "runs/tmp"

    # model size (fits ~5–15M depending)
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1


def _write_json(path: str, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Load train/val blocks
def _load_blocks(path: str) -> torch.Tensor:
    t = torch.load(path, map_location="cpu")
    if not isinstance(t, torch.Tensor) or t.dim() != 2:
        raise ValueError(f"Expected 2D tensor blocks at {path}")
    return t  # [N, block_size+1]


# Sample a random mini-batch for next-token prediction
def _get_batch(blocks: torch.Tensor, batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # blocks: [N, B+1]
    N = blocks.size(0)
    idx = torch.randint(0, N, (batch_size,))
    batch = blocks[idx]  # [bs, B+1]
    x = batch[:, :-1].to(device)
    y = batch[:, 1:].to(device)
    return x, y


# Learning rate schedule
def _cosine_lr(step: int, max_steps: int, warmup: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, (max_steps - warmup))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


# Estimate validation loss by averaging over n_batches random batches
@torch.no_grad()
def eval_loss(model: nn.Module, blocks: torch.Tensor, batch_size: int, device: str, n_batches: int = 50) -> float:
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = _get_batch(blocks, batch_size, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))


# Main training function!!!
def train_one(cfg: TrainConfig) -> str:
    os.makedirs(cfg.out_dir, exist_ok=True)
    _set_seed(cfg.seed)

    train_blocks = _load_blocks(cfg.train_blocks_path)
    val_blocks = _load_blocks(cfg.val_blocks_path)

    gcfg = GPTConfig(
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )
    model = GPTSmall(gcfg).to(cfg.device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # save config
    _write_json(os.path.join(cfg.out_dir, "train_config.json"), asdict(cfg))
    _write_json(os.path.join(cfg.out_dir, "model_config.json"), asdict(gcfg))

    log_path = os.path.join(cfg.out_dir, "log.jsonl")
    def log(row: Dict):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    t0 = time.time()
    best_val = float("inf")
    best_ckpt_path = ""

    for step in range(cfg.max_steps):
        # Compute LR and set optimizer LR
        lr = _cosine_lr(step, cfg.max_steps, cfg.warmup_steps, cfg.lr, cfg.min_lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # Sample batch
        x, y = _get_batch(train_blocks, cfg.batch_size, cfg.device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        # Logging training stats every 50 steps
        if (step + 1) % 50 == 0:
            log({"step": step + 1, "train_loss": float(loss.item()), "lr": lr, "elapsed_s": time.time() - t0})

        # Evaluate every eval_every steps
        if (step + 1) % cfg.eval_every == 0:
            vloss = eval_loss(model, val_blocks, cfg.batch_size, cfg.device)
            log({"step": step + 1, "val_loss": float(vloss)})

            if vloss < best_val:
                best_val = vloss
                best_ckpt_path = os.path.join(cfg.out_dir, "ckpt_best.pt")
                torch.save({"model": model.state_dict(), "gpt_config": asdict(gcfg)}, best_ckpt_path)

        # Save periodic checkpoints every ckpt_every steps
        if (step + 1) % cfg.ckpt_every == 0:
            ckpt_path = os.path.join(cfg.out_dir, f"ckpt_step{step+1}.pt")
            torch.save({"model": model.state_dict(), "gpt_config": asdict(gcfg)}, ckpt_path)

    # final
    final_ckpt = os.path.join(cfg.out_dir, "ckpt_final.pt")
    torch.save({"model": model.state_dict(), "gpt_config": asdict(gcfg)}, final_ckpt)
    return best_ckpt_path or final_ckpt


if __name__ == "__main__":
    # Minimal CLI example: edit here or drive via run_experiment.py
    raise SystemExit("Use run_experiment.py to run end-to-end.")