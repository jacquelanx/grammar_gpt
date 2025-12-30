from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import random

from .grammars import GrammarVocab, _choice


@dataclass(frozen=True)
class PairExample:
    grammatical: str
    ungrammatical: str
    meta: Dict[str, Any]


def flip_verb_number(v: GrammarVocab, verb: str) -> Optional[str]:
    mapping: Dict[str, str] = {}
    for sg, pl in zip(v.iv_sg, v.iv_pl):
        mapping[sg] = pl
        mapping[pl] = sg
    for sg, pl in zip(v.tv_sg, v.tv_pl):
        mapping[sg] = pl
        mapping[pl] = sg
    return mapping.get(verb)


def make_conflict_pair_pp(
    v: GrammarVocab,
    rng: random.Random,
    subject_num: str,
    intervener_num: str,
    p: Optional[str] = None,
    det_subj: Optional[str] = None,
    det_obj: Optional[str] = None,
    noun_subj: Optional[str] = None,
    noun_obj: Optional[str] = None,
) -> PairExample:
    """
    Template: Det Nsubj P Det Nobj V
    Grammatical (for Grammar A): V agrees with subject
    Ungrammatical: flip only V number
    """
    det_subj = det_subj or _choice(rng, v.det)
    det_obj = det_obj or _choice(rng, v.det)
    p = p or _choice(rng, v.p)
    noun_subj = noun_subj or _choice(rng, v.n_sg if subject_num == "sg" else v.n_pl)
    noun_obj = noun_obj or _choice(rng, v.n_sg if intervener_num == "sg" else v.n_pl)

    i = rng.randrange(len(v.iv_sg))
    v_correct = v.iv_sg[i] if subject_num == "sg" else v.iv_pl[i]
    v_wrong = flip_verb_number(v, v_correct)
    assert v_wrong is not None

    gram = f"{det_subj} {noun_subj} {p} {det_obj} {noun_obj} {v_correct}"
    ungram = f"{det_subj} {noun_subj} {p} {det_obj} {noun_obj} {v_wrong}"

    return PairExample(
        grammatical=gram,
        ungrammatical=ungram,
        meta={
            "suite": "conflict_pp",
            "subject_num": subject_num,
            "intervener_num": intervener_num,
            "p": p,
            "noun_subj": noun_subj,
            "noun_obj": noun_obj,
            "verb_correct": v_correct,
            "verb_wrong": v_wrong,
        },
    )


def make_sanity_pair_simple(v: GrammarVocab, rng: random.Random, subject_num: str) -> PairExample:
    det = _choice(rng, v.det)
    noun = _choice(rng, v.n_sg if subject_num == "sg" else v.n_pl)
    i = rng.randrange(len(v.iv_sg))
    v_correct = v.iv_sg[i] if subject_num == "sg" else v.iv_pl[i]
    v_wrong = flip_verb_number(v, v_correct)
    assert v_wrong is not None
    return PairExample(
        grammatical=f"{det} {noun} {v_correct}",
        ungrammatical=f"{det} {noun} {v_wrong}",
        meta={"suite": "sanity_simple", "subject_num": subject_num, "noun": noun, "verb_correct": v_correct},
    )