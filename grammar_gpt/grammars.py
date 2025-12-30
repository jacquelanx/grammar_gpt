from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import random


@dataclass(frozen=True)
class GrammarVocab:
    det: Tuple[str, ...] = ("the", "a")
    n_sg: Tuple[str, ...] = (
        "cat", "dog", "robot", "senator",
        "child", "teacher", "artist", "scientist",
        "pilot", "soldier", "student", "doctor"
    )
    n_pl: Tuple[str, ...] = (
        "cats", "dogs", "robots", "senators",
        "children", "teachers", "artists", "scientists",
        "pilots", "soldiers", "students", "doctors"
    )
    p: Tuple[str, ...] = ("near", "with", "behind", "beside")
    iv_sg: Tuple[str, ...] = ("sleeps", "runs", "laughs", "falls")
    iv_pl: Tuple[str, ...] = ("sleep", "run", "laugh", "fall")
    tv_sg: Tuple[str, ...] = ("sees", "likes", "meets", "follows")
    tv_pl: Tuple[str, ...] = ("see", "like", "meet", "follow")
    rel: Tuple[str, ...] = ("that",)

    def all_word_tokens(self) -> List[str]:
        toks = set()
        for field in self.__dataclass_fields__:
            toks.update(getattr(self, field))
        return sorted(toks)  # ["a", "cat", "cats", "like"...]


def _choice(rng: random.Random, xs: Sequence[str]) -> str:
    return xs[rng.randrange(len(xs))]


@dataclass
class GenConfig:
    p_add_pp: float = 0.6           # prepositional phrase; eg. "on the couch"
    p_add_rc: float = 0.0           # relative clause; eg. "which I saw"
    p_use_transitive: float = 0.0   # no transitives
    max_rc_depth: int = 1           # depth 1: "the dog that barked ran"


class GrammarA:
    """
    Grammar A: hierarchical subject–verb agreement.
    - Pick a subject number (sg/pl).
    - Build an NP with a noun matching that number.
    - Build a VP whose verb agrees with the subject number.
    - Optionally attach a PP to the NP.
    - Optionally attach a relative clause (RC) to the NP.
    """

    def __init__(self, vocab: GrammarVocab):
        self.v = vocab

    def gen_sentence(self, rng: random.Random, cfg: GenConfig) -> str:
        subj_num = rng.choice(["sg", "pl"])
        np = self._gen_np(rng, cfg, num=subj_num, rc_depth=0)  # "the cat with the dogs"
        vp = self._gen_vp(rng, cfg, subj_num=subj_num)         # "sleeps, sleep"
        return f"{np} {vp}"                                    # "the cat with the dogs sleeps"

    # Build a noun phrase that respects number
    def _gen_np(self, rng: random.Random, cfg: GenConfig, num: str, rc_depth: int) -> str:
        det = _choice(rng, self.v.det)                                           # "a, the"
        noun = _choice(rng, self.v.n_sg if num == "sg" else self.v.n_pl)         # "dogs, dog"
        parts = [det, noun]                                                      # ["a", "dog"]
        if rng.random() < cfg.p_add_pp:
            parts.append(self._gen_pp(rng, cfg))                                 # ["a", "dog", "near the cats"]
        if cfg.p_add_rc > 0 and rng.random() < cfg.p_add_rc and rc_depth < cfg.max_rc_depth:
            parts.append(self._gen_rc(rng, cfg, num=num, rc_depth=rc_depth))     # "a dog near the cats that sleeps"
        return " ".join(parts)

    # Intransitive only
    def _gen_vp(self, rng: random.Random, cfg: GenConfig, subj_num: str) -> str:
        v = _choice(rng, self.v.iv_sg if subj_num == "sg" else self.v.iv_pl)     # "sleeps, sleep"
        return v

    def _gen_pp(self, rng: random.Random, cfg: GenConfig) -> str:
        p = _choice(rng, self.v.p)                               # "with, near"
        obj_num = rng.choice(["sg", "pl"])
        obj = self._gen_np(rng, cfg, num=obj_num, rc_depth=0)    # "the cat"
        return f"{p} {obj}"

    def _gen_rc(self, rng: random.Random, cfg: GenConfig, num: str, rc_depth: int) -> str:
        rel = _choice(rng, self.v.rel)                               # "that, which"
        if rng.random() < cfg.p_use_transitive:                      # "chase, chases"
            verb = _choice(rng, self.v.tv_sg if num == "sg" else self.v.tv_pl)
            obj_num = rng.choice(["sg", "pl"])
            obj = self._gen_np(rng, cfg, num=obj_num, rc_depth=0)    # "a cat"
            return f"{rel} {verb} {obj}"                             # "that chases a cat"
        else:
            verb = _choice(rng, self.v.iv_sg if num == "sg" else self.v.iv_pl)
            return f"{rel} {verb}"                                   # "that sleeps"

    def is_grammatical(self, sent: str) -> bool:
        toks = sent.strip().split()
        if len(toks) < 3:
            return False

        all_vocab = set(self.v.all_word_tokens())
        if any(t not in all_vocab for t in toks):
            return False

        # helpers
        det_set = set(self.v.det)
        n_sg_set = set(self.v.n_sg)
        n_pl_set = set(self.v.n_pl)
        p_set = set(self.v.p)
        rel_set = set(self.v.rel)

        iv_sg_set = set(self.v.iv_sg)
        iv_pl_set = set(self.v.iv_pl)
        tv_sg_set = set(self.v.tv_sg)
        tv_pl_set = set(self.v.tv_pl)

        verb_any_set = iv_sg_set | iv_pl_set | tv_sg_set | tv_pl_set
        # eg. {"sleeps","runs","sleep","run","sees","likes","see","like"}

        def _noun_num(noun_tok: str) -> str | None:
            if noun_tok in n_sg_set:
                return "sg"
            if noun_tok in n_pl_set:
                return "pl"
            return None

        def _verb_num(verb_tok: str) -> str | None:
            if verb_tok in (iv_sg_set | tv_sg_set):
                return "sg"
            if verb_tok in (iv_pl_set | tv_pl_set):
                return "pl"
            return None

        # Parse NP of the form:
        #   NP -> DET N (PP)* (RC)?
        #   PP -> P NP
        #   RC -> REL (IV | TV NP)
        #
        # We return (new_index, head_num) where head_num is the number of the head noun
        def parse_np(i: int) -> Tuple[int, str] | None:
            # Expect DET
            if i >= len(toks) or toks[i] not in det_set:
                return None
            det = toks[i]  # "the"
            i += 1

            # Expect NOUN
            if i >= len(toks):
                return None
            noun = toks[i]  # "cat, cats"
            head_num = _noun_num(noun)
            if head_num is None:
                return None
            i += 1

            # 0+ PP: P NP
            # Example: "the cat with the dogs near a robot ..."
            while i < len(toks) and toks[i] in p_set:
                p = toks[i]  # "with, near"
                i += 1
                # After a preposition, we must see another NP
                res = parse_np(i)
                if res is None:
                    return None
                i, _ = res  # PP object NP number doesn't affect agreement for Grammar A

            # Optional RC at the end of NP
            # Example: "the cat ... that sleeps" or "the cats ... that see a dog"
            if i < len(toks) and toks[i] in rel_set:
                res = parse_rc(i, head_num=head_num)
                if res is None:
                    return None
                i = res

            return (i, head_num)

        # Parse RC of the form:
        #   RC -> REL IV
        #      | REL TV NP
        #
        # In Grammar A, the RC verb agrees with the head noun of the NP it attaches to
        def parse_rc(i: int, head_num: str) -> int | None:
            # Expect REL
            if i >= len(toks) or toks[i] not in rel_set:
                return None
            rel = toks[i]  # "that"
            i += 1

            # Expect VERB (intransitive or transitive)
            if i >= len(toks) or toks[i] not in verb_any_set:
                return None
            verb = toks[i]  # "sleeps" / "sleep" / "sees" / "see" ...
            verb_num = _verb_num(verb)
            if verb_num is None:
                return None

            # Enforce agreement inside the RC
            # Example: "the cat that sleep" should fail here
            if verb_num != head_num:
                return None
            i += 1

            # If the RC verb is transitive, we must parse an object NP
            # Example: "the cat that sees the dogs ..."
            if verb in (tv_sg_set | tv_pl_set):
                res = parse_np(i)
                if res is None:
                    return None
                i, _ = res

            return i

        # Main parse: sentence -> NP VP 

        # Parse the subject NP starting at token 0
        res_np = parse_np(0)
        if res_np is None:
            return False
        i, subj_num = res_np

        # After NP, expect a main-clause verb; accept IV or TV (and if TV, require an object NP).
        if i >= len(toks) or toks[i] not in verb_any_set:
            return False
        verb = toks[i]  # "sleeps, sleep, sees, see"
        verb_num = _verb_num(verb)
        if verb_num is None:
            return False

        # Enforce main clause subject–verb agreement (hierarchical: agrees with head noun)
        # eg. "the cats near the dog sleeps" should fail here
        if verb_num != subj_num:
            return False
        i += 1

        # If main verb is transitive, parse the object NP
        # Example: "the cat sees the dogs"
        if verb in (tv_sg_set | tv_pl_set):
            res_obj = parse_np(i)
            if res_obj is None:
                return False
            i, _ = res_obj

        # Must consume all tokens exactly (no leftover garbage)
        # eg. "the cat sleeps near the dog" should be rejected under this grammar
        # because PPs attach to NPs here, not to VPs
        if i != len(toks):
            return False

        return True
    

class GrammarB:
    """
    Grammar B: competing grammar with nearest-noun agreement.
    Verb agrees with the last noun before the verb (often the PP object).
    Structural shape (same surface syntax as A, different agreement rule):
    - Sentence: NP VP
    - NP: DET NOUN (PP)* (RC)?   
    - PP: P NP
    - RC: REL (IV | TV NP)
    """

    def __init__(self, vocab: GrammarVocab):
        self.v = vocab

    def gen_sentence(self, rng: random.Random, cfg: GenConfig) -> str:
        det = _choice(rng, self.v.det)
        head_num = rng.choice(["sg", "pl"])
        head_n = _choice(rng, self.v.n_sg if head_num == "sg" else self.v.n_pl)
        toks = [det, head_n]  # ["the", "cat"]

        if rng.random() < cfg.p_add_pp:
            p = _choice(rng, self.v.p)
            od = _choice(rng, self.v.det)
            onum = rng.choice(["sg", "pl"])
            on = _choice(rng, self.v.n_sg if onum == "sg" else self.v.n_pl)
            toks += [p, od, on]  # ["the", "cat", "with", "a", "dogs"]

        nearest = next(t for t in reversed(toks) if t in self.v.n_sg or t in self.v.n_pl)
        nearest_num = "sg" if nearest in self.v.n_sg else "pl"
        v = _choice(rng, self.v.iv_sg if nearest_num == "sg" else self.v.iv_pl)  # nearest-noun agreement
        toks.append(v)
        return " ".join(toks) # "the cat with a dogs sleep"

    def is_grammatical(self, sent: str) -> bool:
        toks = sent.strip().split()
        if len(toks) < 3:
            return False

        all_vocab = set(self.v.all_word_tokens())
        if any(t not in all_vocab for t in toks):
            return False

        det_set = set(self.v.det)
        n_sg_set = set(self.v.n_sg)
        n_pl_set = set(self.v.n_pl)
        p_set = set(self.v.p)
        rel_set = set(self.v.rel)

        iv_sg_set = set(self.v.iv_sg)
        iv_pl_set = set(self.v.iv_pl)
        tv_sg_set = set(self.v.tv_sg)
        tv_pl_set = set(self.v.tv_pl)

        verb_any_set = iv_sg_set | iv_pl_set | tv_sg_set | tv_pl_set  # all verbs

        def _noun_num(noun_tok: str) -> str | None:
            if noun_tok in n_sg_set:
                return "sg"
            if noun_tok in n_pl_set:
                return "pl"
            return None

        def _verb_num(verb_tok: str) -> str | None:
            if verb_tok in (iv_sg_set | tv_sg_set):
                return "sg"
            if verb_tok in (iv_pl_set | tv_pl_set):
                return "pl"
            return None

        # Structure checks:
        # - Sentence starts with a valid NP
        # - Then has a VP verb (optionally transitive with an object NP)
        # - No extra tokens at the end
        # But the main verb agreement check uses nearest noun before verb.
        #
        # Parse NP of the form:
        #   NP -> DET N (PP)* (RC)?
        # We return (new_index, head_num, last_noun_num)
        #   head_num: number of the head noun of this NP ("cat" in "the cat with the dogs")
        #   last_noun_num: number of the last noun inside this NP (often PP object noun)
        def parse_np(i: int) -> Tuple[int, str, str] | None:
            # Expect DET
            if i >= len(toks) or toks[i] not in det_set:
                return None
            det = toks[i]  # "the"
            i += 1

            # Expect NOUN
            if i >= len(toks):
                return None
            noun = toks[i]  # "cat, cats"
            head_num = _noun_num(noun)
            if head_num is None:
                return None
            last_noun_num = head_num  # last noun seen so far is the head noun
            i += 1

            # 0+ PP: P NP
            # eg. "the cat with the dogs near a robot ..."
            while i < len(toks) and toks[i] in p_set:
                p = toks[i]  # "with, near"
                i += 1

                res = parse_np(i)  # parse object NP immediately after P
                if res is None:
                    return None
                i, _obj_head_num, obj_last_noun_num = res

                # Update "nearest noun" candidate using the last noun inside the PP object NP
                last_noun_num = obj_last_noun_num

            # Optional RC at end of NP
            # eg. "the cat ... that sleeps"
            if i < len(toks) and toks[i] in rel_set:
                res = parse_rc(i, rc_head_num=head_num)
                if res is None:
                    return None
                i = res

            return (i, head_num, last_noun_num)

        # Parse RC of the form:
        #   RC -> REL IV
        #      | REL TV NP
        #
        # We still enforce RC verb agreement with the RC head noun
        # so RCs remain internally consistent (systematic).
        def parse_rc(i: int, rc_head_num: str) -> int | None:
            # Expect REL
            if i >= len(toks) or toks[i] not in rel_set:
                return None
            rel = toks[i]  # "that"
            i += 1

            # Expect VERB
            if i >= len(toks) or toks[i] not in verb_any_set:
                return None
            verb = toks[i]  # "sleeps, sleep, sees, see"
            verb_num = _verb_num(verb)
            if verb_num is None:
                return None

            # RC agreement (still hierarchical with RC head noun)
            if verb_num != rc_head_num:
                return None
            i += 1

            # If RC verb is transitive, require object NP
            if verb in (tv_sg_set | tv_pl_set):
                res = parse_np(i)
                if res is None:
                    return None
                i, _obj_head_num, _obj_last_noun_num = res

            return i

        # Main parse: sentence -> NP VP 

        # Parse the subject NP starting at token 0
        res_np = parse_np(0)
        if res_np is None:
            return False
        i, subj_head_num, subj_last_noun_num = res_np
        # eg. "the cat with the dogs sleep"
        #   subj_head_num = "sg"  (cat)
        #   subj_last_noun_num = "pl" (dogs)  <-- nearest noun inside NP

        # Expect a main-clause verb immediately after NP
        if i >= len(toks) or toks[i] not in verb_any_set:
            return False
        verb = toks[i]  # main verb
        verb_num = _verb_num(verb)
        if verb_num is None:
            return False

        # Nearest-noun agreement for the main clause
        # eg. "the cat with the dogs sleep" is grammatical in B:
        #   nearest noun before verb = "dogs" (pl) -> verb must be plural -> "sleep"
        if verb_num != subj_last_noun_num:
            return False
        i += 1

        # If main verb is transitive, require object NP
        if verb in (tv_sg_set | tv_pl_set):
            res_obj = parse_np(i)
            if res_obj is None:
                return False
            i, _obj_head_num, _obj_last_noun_num = res_obj

        # Must consume all tokens (no VP-attached PPs, no trailing tokens)
        if i != len(toks):
            return False

        return True