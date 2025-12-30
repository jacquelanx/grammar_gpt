from __future__ import annotations

from typing import Dict

from .grammars import GrammarVocab
from .scoring import PairResult


# When the model gets a pair “wrong,” what kind of mistake was it making?
def error_typology_pair(r: PairResult, v: GrammarVocab) -> str:
    if r.correct == 1:
        return "OK"
    g = r.grammatical.split()
    u = r.ungrammatical.split()
    diff = [i for i, (a, b) in enumerate(zip(g, u)) if a != b]
    if len(diff) != 1:
        return "E_other_nonminimal"
    vi = diff[0]
    # nearest noun before verb in ungrammatical choice
    nearest = next((t for t in reversed(u[:vi]) if t in v.n_sg or t in v.n_pl), None)
    if nearest is None:
        return "E_parse_fail"
    nearest_num = "sg" if nearest in v.n_sg else "pl"
    chosen_verb = u[vi]
    if nearest_num == "sg" and chosen_verb in (v.iv_sg + v.tv_sg):
        return "E1_nearest_noun_heuristic"
    if nearest_num == "pl" and chosen_verb in (v.iv_pl + v.tv_pl):
        return "E1_nearest_noun_heuristic"
    if chosen_verb in (v.iv_sg + v.tv_sg):
        return "E2_prefers_singular_form"
    if chosen_verb in (v.iv_pl + v.tv_pl):
        return "E2_prefers_plural_form"
    return "E3_other"