from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenization import SimpleTokenizer
from .grammars import GrammarA, GrammarVocab, GenConfig
from .pairs import make_conflict_pair_pp


# Attention probe
@dataclass
class AttnProbeSpec:
    n_examples: int = 500   # generate 500 test sentences
    seed: int = 2024


@torch.no_grad()
def attention_probe_subject_vs_intervener(
    model: nn.Module,
    tok: SimpleTokenizer,
    device: str,
    v: GrammarVocab,
    spec: AttnProbeSpec,
) -> Dict[str, Any]:
    """
    Requires model(x) -> (logits, attns) where attns is list[n_layers] of [B, heads, T, T].
    B = batch size; head = # attention heads; T = word position (from); T = word position (to)
    Assumes word-level tokenizer so indices align with tokens.
    """
    if tok.mode != "word":
        return {"error": "Attention probe expects word-level tokenizer."}

    # Make examples 
    rng = random.Random(spec.seed)
    examples = []
    for i in range(spec.n_examples):
        subj_num, int_num = ("sg", "pl") if i % 2 == 0 else ("pl", "sg")
        ex = make_conflict_pair_pp(v, rng, subj_num, int_num)
        sent = ex.grammatical
        # Template indices: Det(0) Nsubj(1) P(2) Det(3) Nobj(4) V(5)
        # How much attention goes from word 5 → word 1 vs word 4?
        examples.append((sent, 1, 4, 5))

    ratios = []
    for sent, subj_i, intv_i, verb_i in examples:
        ids = tok.encode(sent, add_eos=True)
        if len(ids) < 2:
            continue
        x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
        
        # Return predictions in form (logits, attns)
        out = model(x)
        if not (isinstance(out, (tuple, list)) and len(out) >= 2):
            return {"error": "Model must return (logits, attns) for attention probe."}
        # Get attention matrix
        attns = out[1]
        if verb_i >= x.size(1):
            continue
        layer_ratios = []
        for layer_attn in attns:                                        # attn[1] = attentions @ layer 1 with shape [B, heads, T, T]
            a = layer_attn[0]                                           # [heads, T, T]
            num = a[:, verb_i, subj_i].detach().float().cpu().numpy()   # for EVERY attention head, how much the verb looks at the subject
            den = a[:, verb_i, intv_i].detach().float().cpu().numpy()   # for EVERY attention head, how much the verb looks at the wrong noun
            layer_ratios.append((num + 1e-9) / (den + 1e-9))            # attention accuracy ratio
        ratios.append(layer_ratios)

    if not ratios:
        return {"n": 0, "note": "No usable examples. Check model outputs and tokenizer alignment."}

    # Average across all sentences
    arr = np.array(ratios)  # [N, layers, heads]
    return {
        "n": int(arr.shape[0]),
        "mean_ratio_by_layer_head": arr.mean(axis=0).tolist(),
        "note": "ratio = attn(verb->subject_head) / attn(verb->intervener)",
    }


# Probing classifier
@dataclass
class ProbeSpec:
    n_train: int = 5000
    n_test: int = 1000
    seed: int = 777


class HiddenExtractor:
    """
    Given token ids x, give me the hidden vectors.
    Requires either:
      - model.forward_hidden(x) -> [B,T,D], OR
      - model(x, return_hidden=True) -> (logits, hidden)
    """
    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_hidden"):       # tries attempt 1
            return self.model.forward_hidden(x)         # type: ignore[attr-defined]
        try:
            out = self.model(x, return_hidden=True)     # type: ignore[misc]
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                return out[1]
        except TypeError:
            pass
        raise RuntimeError("Model must expose hidden states (forward_hidden or return_hidden=True).")


# Very simple: given the internal vector, can I tell whether the subject was singular or plural?
class LogisticProbe(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.lin = nn.Linear(d_in, 2)   # 0 = singular subject, 1 = plural

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


# Pad short sentences to match long sentence length
def _pad(seqs: List[List[int]], pad_id: int) -> torch.Tensor:
    mx = max(len(s) for s in seqs)
    out = torch.full((len(seqs), mx), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return out


# Ignore padding and find the real end of each sentence
def _last_nonpad_idx(x: torch.Tensor, pad_id: int) -> torch.Tensor:
    # last index where token != pad_id; if all pad, return 0
    idxs = []
    for row in x.detach().cpu().numpy():
        nz = np.where(row != pad_id)[0]
        idxs.append(int(nz[-1]) if len(nz) else 0)
    return torch.tensor(idxs, dtype=torch.long, device=x.device)


def run_probe_subject_number(
    model: nn.Module,
    tok: SimpleTokenizer,
    device: str,
    grammarA: GrammarA,
    spec: ProbeSpec,
) -> Dict[str, Any]:
    rng = random.Random(spec.seed)
    gen_cfg = GenConfig(p_add_pp=0.6, p_add_rc=0.0)

    def make_dataset(n: int) -> Tuple[List[List[int]], List[int]]:
        X, y = [], []
        for _ in range(n):
            s = grammarA.gen_sentence(rng, gen_cfg)
            noun = s.split()[1]
            label = 0 if noun in grammarA.v.n_sg else 1
            X.append(tok.encode(s, add_eos=True))
            y.append(label)
        return X, y

    Xtr, ytr = make_dataset(spec.n_train)   # 5000
    Xte, yte = make_dataset(spec.n_test)    # 1000

    pad_id = tok.stoi[tok.eos_token]
    xtr = _pad(Xtr, pad_id).to(device)
    xte = _pad(Xte, pad_id).to(device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    yte_t = torch.tensor(yte, dtype=torch.long, device=device)

    extractor = HiddenExtractor(model)
    with torch.no_grad():
        h = extractor.hidden(xtr[:, :-1])   # hidden state tensor shape [B, T, D]
        d = h.size(-1)                      # Grabs D, the hidden dimension

    # Optimizes probe
    probe = LogisticProbe(d).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.0)

    # Train probe for 10 steps
    for _ in range(10):
        probe.train()
        opt.zero_grad()
        h = extractor.hidden(xtr[:, :-1])                           # get hidden state
        idx = _last_nonpad_idx(xtr[:, :-1], pad_id)                 # find last real token index per sentence
        feats = h[torch.arange(h.size(0), device=device), idx, :]   # take hidden state at end of every sentence
        logits = probe(feats)                                       # logits are scores for 0 or 1
        loss = F.cross_entropy(logits, ytr_t)
        loss.backward()
        opt.step()

    # Evaluate on test set
    probe.eval()
    with torch.no_grad():
        h = extractor.hidden(xte[:, :-1])
        idx = _last_nonpad_idx(xte[:, :-1], pad_id)
        feats = h[torch.arange(h.size(0), device=device), idx, :]
        pred = torch.argmax(probe(feats), dim=-1)
        acc = (pred == yte_t).float().mean().item()     # get guess accuracy

    return {"probe_acc_subject_number": float(acc), "n_train": spec.n_train, "n_test": spec.n_test}