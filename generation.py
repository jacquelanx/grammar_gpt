from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from .tokenization import SimpleTokenizer
from .grammars import GrammarA, GrammarB


@dataclass
class GenEvalConfig:
    prompts: Tuple[str, ...] = ("the cat", "the cats", "the cat near", "the dogs near the cat")
    n_samples_per_prompt: int = 200     # prompts are above: how many samples generated per prompt?
    max_new_tokens: int = 12            # generation length
    temperature: float = 1.0            # 0 to 1: top tokens dominate; 1: take logits as is; >1: probabilities more even so more random
    top_p: float = 0.95                 # only sample from top 95% probability
    seed: int = 999                     # random seed


@torch.no_grad()
def _sample_next_token(logits: torch.Tensor, temperature: float, top_p: float, rng: np.random.Generator) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())                     # take most likely token
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()    
    idx = np.argsort(probs)[::-1]                                   # sort tokens by probability
    sorted_probs = probs[idx]
    cumsum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumsum, top_p) + 1                     # keep the top 95% tokens
    idx = idx[:cutoff]
    sorted_probs = sorted_probs[:cutoff]
    sorted_probs = sorted_probs / sorted_probs.sum()                # renormalize after cutoff
    return int(rng.choice(idx, p=sorted_probs))                     # randomly sample a token


@torch.no_grad()
def generate(
    model: nn.Module,
    tok: SimpleTokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    rng: np.random.Generator,
) -> str:
    ids = tok.encode(prompt, add_eos=False)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    for _ in range(max_new_tokens):                                     # loop up to max new tokens
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        next_logits = logits[0, -1]
        nxt = _sample_next_token(next_logits, temperature, top_p, rng)  # sample next token
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1) # add to sequence
        if tok.itos.get(nxt, "") == tok.eos_token:
            break
    return tok.decode(x[0].tolist())


def run_generation_suite(
    model: nn.Module,
    tok: SimpleTokenizer,
    device: str,
    cfg: GenEvalConfig,
    grammarA: GrammarA,
    grammarB: GrammarB,
) -> Dict[str, Any]:
    rng = np.random.default_rng(cfg.seed)
    rows: List[Dict[str, Any]] = []
    for p in cfg.prompts:
        for _ in range(cfg.n_samples_per_prompt):
            s = generate(model, tok, p, device, cfg.max_new_tokens, cfg.temperature, cfg.top_p, rng)
            toks = [t for t in s.split() if t != tok.eos_token]
            s2 = " ".join(toks)
            rows.append(
                {
                    "prompt": p,
                    "sent": s2,
                    "okA": grammarA.is_grammatical(s2),
                    "okB": grammarB.is_grammatical(s2),
                }
            )
    # Aggregate stats
    pctA = sum(r["okA"] for r in rows) / max(len(rows), 1)
    pctB = sum(r["okB"] for r in rows) / max(len(rows), 1)
    return {"n": len(rows), "parseable_A": pctA, "parseable_B": pctB, "examples": rows[:25]}