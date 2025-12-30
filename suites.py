from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import random

from .grammars import GrammarVocab
from .pairs import PairExample, make_conflict_pair_pp, make_sanity_pair_simple


@dataclass
class EvalSpec:
    seed: int = 123                     # random seed
    n_conflict: int = 2000              # number of conflict cases
    n_sanity: int = 1000                # number of sanity checks
    n_holdout_lexrole: int = 1000       # lexical generalization tests

    conflict_p_choices: Tuple[str, ...] = ("near", "with")
    sanity_sg_prob: float = 0.5

    # lexical-role holdout: subject noun(s) that were never used as subjects in training
    holdout_subject_nouns: Tuple[str, ...] = ("senator", "senators")


# Builds conflict pairs
def build_conflict_suite(v: GrammarVocab, spec: EvalSpec) -> List[PairExample]:
    rng = random.Random(spec.seed + 1)
    out: List[PairExample] = []
    for i in range(spec.n_conflict):
        subj_num, int_num = ("sg", "pl") if i % 2 == 0 else ("pl", "sg")
        p = spec.conflict_p_choices[i % len(spec.conflict_p_choices)]
        out.append(make_conflict_pair_pp(v, rng, subj_num, int_num, p=p))
    return out


# Simple subject–verb agreement pairs for validation
def build_sanity_suite(v: GrammarVocab, spec: EvalSpec) -> List[PairExample]:
    rng = random.Random(spec.seed + 2)
    out: List[PairExample] = []
    for _ in range(spec.n_sanity):
        subj_num = "sg" if rng.random() < spec.sanity_sg_prob else "pl"
        out.append(make_sanity_pair_simple(v, rng, subj_num))
    return out


# Builds conflict pairs with witheld data to test generalization
def build_holdout_lexrole_suite(v: GrammarVocab, spec: EvalSpec) -> List[PairExample]:
    rng = random.Random(spec.seed + 3)
    held_sg = [n for n in spec.holdout_subject_nouns if n in v.n_sg]
    held_pl = [n for n in spec.holdout_subject_nouns if n in v.n_pl]
    if not held_sg and not held_pl:
        raise ValueError("holdout_subject_nouns must overlap vocab nouns.")

    out: List[PairExample] = []
    for i in range(spec.n_holdout_lexrole):
        if i % 2 == 0 and held_sg:
            subj_num, subj_noun, int_num = "sg", rng.choice(held_sg), "pl"
        else:
            subj_num, subj_noun, int_num = "pl", rng.choice(held_pl or list(v.n_pl)), "sg"
        ex = make_conflict_pair_pp(v, rng, subj_num, int_num, noun_subj=subj_noun)
        ex.meta["suite"] = "holdout_lexrole"
        ex.meta["heldout_subject"] = True
        out.append(ex)
    return out
