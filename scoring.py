from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenization import SimpleTokenizer
from .pairs import PairExample


@torch.no_grad()
def sentence_nll(model: nn.Module, tok: SimpleTokenizer, sentence: str, device: str) -> Tuple[float, int]:
    ids = tok.encode(sentence, add_eos=True) # ["the", "cat", "sleeps"] --> [0, 1, 2]
    if len(ids) < 2:
        return 0.0, 0
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0) # inputs
    y = torch.tensor(ids[1:], dtype=torch.long, device=device).unsqueeze(0)  # targets

    out = model(x)
    logits = out[0] if isinstance(out, (tuple, list)) else out              # allow (logits, ...)
    logp = F.log_softmax(logits, dim=-1)
    nll = F.nll_loss(logp.view(-1, logp.size(-1)), y.view(-1), reduction="sum").item()
    return float(nll), int(y.numel())


@torch.no_grad()
def sentence_nnll(model: nn.Module, tok: SimpleTokenizer, sentence: str, device: str) -> float:
    nll, T = sentence_nll(model, tok, sentence, device)
    return nll / max(T, 1) # length-normalize it


@dataclass
class PairResult:
    grammatical: str
    ungrammatical: str
    nnll_gram: float
    nnll_ungram: float
    delta: float
    correct: int
    meta: Dict[str, Any]


def score_pair(model: nn.Module, tok: SimpleTokenizer, ex: PairExample, device: str) -> PairResult:
    g = sentence_nnll(model, tok, ex.grammatical, device)   # nll of grammatical sentence
    u = sentence_nnll(model, tok, ex.ungrammatical, device) # nll of ungrammatical sentence
    delta = u - g                                           # difference b/t the nlls
    return PairResult(
        grammatical=ex.grammatical,
        ungrammatical=ex.ungrammatical,
        nnll_gram=g,
        nnll_ungram=u,
        delta=delta,
        correct=1 if delta > 0 else 0,
        meta=ex.meta,
    )