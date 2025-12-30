"""
Generates A/B/C datasets + splits + holdout knobs.
Inside spec.out_dir (default "data"), it writes:
    - tokenizer.json: the word-level vocabulary used for tokenizing
    - A_train_blocks.pt, A_val_blocks.pt: GrammarA-only token blocks
    - B_train_blocks.pt, B_val_blocks.pt: mostly GrammarA with a small fraction from GrammarB (default 5%)
    - C_train_blocks.pt, C_val_blocks.pt: GrammarB-only token blocks
    - data_spec.json: the exact settings used
"""
from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import torch

from .grammars import GrammarA, GrammarB, GrammarVocab, GenConfig
from .tokenization import SimpleTokenizer


@dataclass
class DataSpec:
    seed: int = 123
    out_dir: str = "data"

    # total examples
    n_train: int = 200_000
    n_val: int = 10_000

    # mixing for condition B (Grammar A + tiny "corpus"/other grammar)
    mix_b_fraction_from_B: float = 0.05  # 5% GrammarB, 95% GrammarA

    # sentence generation settings (keep comparable across conditions)
    gen_cfg_A: GenConfig = GenConfig(p_add_pp=0.7, p_add_rc=0.0, p_use_transitive=0.0)
    gen_cfg_B: GenConfig = GenConfig(p_add_pp=0.7, p_add_rc=0.0, p_use_transitive=0.0)

    # sequence/block settings
    block_size: int = 64
    add_eos: bool = True

    # lexical-role holdout (optional): keep nouns from appearing as SUBJECT in training
    # (you’ll still evaluate them as subjects in Suite 3 holdout_lexrole)
    holdout_subject_nouns: Tuple[str, ...] = ("senator", "senators")


def _write_json(path: str, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _make_tokenizer(v: GrammarVocab) -> SimpleTokenizer:
    # Word-level vocab from grammar terminals (+ specials added in tokenizer)
    return SimpleTokenizer(vocab=v.all_word_tokens(), mode="word")


def _subject_noun(sent: str) -> str:
    toks = sent.split()
    return toks[1] if len(toks) >= 2 else ""


def _generate_sentences_condition(
    cond: str,
    n: int,
    rng: random.Random,
    grammarA: GrammarA,
    grammarB: GrammarB,
    spec: DataSpec,
) -> List[str]:
    """
    cond:
      - "A": GrammarA only
      - "B": mixture (mostly A + small B fraction)
      - "C": GrammarB only
    """
    out: List[str] = []
    while len(out) < n:
        if cond == "A":
            s = grammarA.gen_sentence(rng, spec.gen_cfg_A)
            # apply lexical-role holdout (subject nouns excluded)
            if _subject_noun(s) in spec.holdout_subject_nouns:
                continue
            out.append(s)

        elif cond == "C":
            s = grammarB.gen_sentence(rng, spec.gen_cfg_B)
            # also exclude heldout subjects to keep comparable, though B sentences may violate this notion
            if _subject_noun(s) in spec.holdout_subject_nouns:
                continue
            out.append(s)

        elif cond == "B":
            use_B = (rng.random() < spec.mix_b_fraction_from_B)
            if use_B:
                s = grammarB.gen_sentence(rng, spec.gen_cfg_B)
            else:
                s = grammarA.gen_sentence(rng, spec.gen_cfg_A)
            if _subject_noun(s) in spec.holdout_subject_nouns:
                continue
            out.append(s)
        else:
            raise ValueError(f"Unknown condition: {cond}")
    return out


def _sentences_to_blocks(
    sentences: List[str],
    tok: SimpleTokenizer,
    block_size: int,
    add_eos: bool,
) -> torch.Tensor:
    """
    Turn a list of sentences into a long token stream, then chunk into blocks for LM training:
      x = tokens[i:i+block_size]
      y = tokens[i+1:i+block_size+1]
    """
    ids: List[int] = []
    for s in sentences:
        ids.extend(tok.encode(s, add_eos=add_eos))
    stream = torch.tensor(ids, dtype=torch.long)
    # ensure at least block_size+1
    if stream.numel() < block_size + 1:
        raise ValueError("Not enough tokens to form a single block.")
    # truncate to multiple of block_size+1? we'll just make as many blocks as possible
    n_blocks = (stream.numel() - 1) // block_size
    stream = stream[: n_blocks * block_size + 1]
    # shape [n_blocks, block_size+1]
    blocks = stream.unfold(0, block_size + 1, block_size)
    # blocks: [n_blocks, block_size+1]
    return blocks.contiguous()


def build_datasets(spec: DataSpec) -> Dict[str, str]:
    os.makedirs(spec.out_dir, exist_ok=True)
    rng = random.Random(spec.seed)

    vocab = GrammarVocab()
    tok = _make_tokenizer(vocab)

    grammarA = GrammarA(vocab)
    grammarB = GrammarB(vocab)

    # Save tokenizer for eval
    tok_path = os.path.join(spec.out_dir, "tokenizer.json")
    _write_json(tok_path, {"mode": tok.mode, "vocab": tok.vocab, "unk_token": tok.unk_token, "eos_token": tok.eos_token})

    # Generate and save each condition
    paths: Dict[str, str] = {"tokenizer": tok_path}

    for cond in ["A", "B", "C"]:
        train_sents = _generate_sentences_condition(cond, spec.n_train, rng, grammarA, grammarB, spec)
        val_sents = _generate_sentences_condition(cond, spec.n_val, rng, grammarA, grammarB, spec)

        train_blocks = _sentences_to_blocks(train_sents, tok, spec.block_size, spec.add_eos)
        val_blocks = _sentences_to_blocks(val_sents, tok, spec.block_size, spec.add_eos)

        # Save as .pt
        train_path = os.path.join(spec.out_dir, f"{cond}_train_blocks.pt")
        val_path = os.path.join(spec.out_dir, f"{cond}_val_blocks.pt")
        torch.save(train_blocks, train_path)
        torch.save(val_blocks, val_path)

        paths[f"{cond}_train"] = train_path
        paths[f"{cond}_val"] = val_path

    # Save spec for reproducibility
    _write_json(os.path.join(spec.out_dir, "data_spec.json"), asdict(spec))

    return paths


if __name__ == "__main__":
    # basic CLI-less entrypoint (edit spec here if you want)
    spec = DataSpec()
    paths = build_datasets(spec)
    print("Built datasets:", paths)