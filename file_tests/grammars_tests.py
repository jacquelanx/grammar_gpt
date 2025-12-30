import random
from grammars import GrammarA, GrammarB, GrammarVocab, GenConfig  


# ----------------------------
# Grammar A tests
# ----------------------------

def test_generator_outputs_are_grammatical_A():
    v = GrammarVocab()
    g = GrammarA(v)

    rng = random.Random(0)
    cfg = GenConfig(p_add_pp=0.9, p_add_rc=0.0, p_use_transitive=0.0)

    for _ in range(200):
        s = g.gen_sentence(rng, cfg)
        assert g.is_grammatical(s), f"[A] Generated but rejected: {s}"


def test_rejects_subject_verb_disagreement_simple_A():
    v = GrammarVocab()
    g = GrammarA(v)

    assert g.is_grammatical("the cat sleeps")
    assert g.is_grammatical("the cats sleep")

    assert not g.is_grammatical("the cat sleep")      # sg noun + pl verb
    assert not g.is_grammatical("the cats sleeps")    # pl noun + sg verb


def test_accepts_pp_inside_np_A():
    v = GrammarVocab()
    g = GrammarA(v)

    assert g.is_grammatical("the cat with the dogs sleeps")
    assert g.is_grammatical("the cats near a dog sleep")
    assert g.is_grammatical("the cat with the dogs near a robot runs")


def test_rejects_pp_after_vp_A():
    v = GrammarVocab()
    g = GrammarA(v)

    assert not g.is_grammatical("the cat sleeps with the dogs")
    assert not g.is_grammatical("the cats run near a dog")


def test_rejects_unknown_tokens_and_bad_starts_A():
    v = GrammarVocab()
    g = GrammarA(v)

    assert not g.is_grammatical("cat sleeps")               # missing det
    assert not g.is_grammatical("the green cat sleeps")     # "green" not in vocab
    assert not g.is_grammatical("the cat quickly sleeps")   # "quickly" not in vocab


def test_transitives_main_clause_A_if_supported():
    v = GrammarVocab()
    g = GrammarA(v)

    # If your GrammarA recognizer supports transitive main verbs, keep these.
    # If not, delete this test.
    assert g.is_grammatical("the cat sees the dog")
    assert g.is_grammatical("the cats see the dog")

    assert not g.is_grammatical("the cats sees the dog")    # pl subj + sg verb
    assert not g.is_grammatical("the cat see the dog")      # sg subj + pl verb

    assert not g.is_grammatical("the cat sees")             # TV needs object NP


# ----------------------------
# Grammar B tests (beneath A)
# ----------------------------

def test_generator_outputs_are_grammatical_B():
    v = GrammarVocab()
    g = GrammarB(v)

    rng = random.Random(0)
    cfg = GenConfig(p_add_pp=0.9, p_add_rc=0.0, p_use_transitive=0.0)

    for _ in range(200):
        s = g.gen_sentence(rng, cfg)
        assert g.is_grammatical(s), f"[B] Generated but rejected: {s}"


def test_nearest_noun_agreement_signature_B():
    v = GrammarVocab()
    g = GrammarB(v)

    # No PP: nearest noun is the head noun, so B looks like A here
    assert g.is_grammatical("the cat sleeps")
    assert g.is_grammatical("the cats sleep")
    assert not g.is_grammatical("the cat sleep")
    assert not g.is_grammatical("the cats sleeps")

    # With PP: nearest noun is the PP object noun, so B differs from A
    # "the cat with the dogs sleep" -> nearest noun before verb = "dogs" (pl) -> verb must be pl ("sleep")
    assert g.is_grammatical("the cat with the dogs sleep")
    assert not g.is_grammatical("the cat with the dogs sleeps")  # would be A-like, not B-like

    # "the cats near a dog runs" -> nearest noun before verb = "dog" (sg) -> verb must be sg ("runs")
    assert g.is_grammatical("the cats near a dog runs")
    assert not g.is_grammatical("the cats near a dog run")       # would be A-like, not B-like


def test_accepts_pp_inside_np_B_and_tracks_nearest_noun_B():
    v = GrammarVocab()
    g = GrammarB(v)

    # Multiple PPs: nearest noun is the LAST noun in the whole NP
    # NP = "the cat with the dogs near a robot" -> nearest noun = "robot" (sg) -> main verb must be sg
    assert g.is_grammatical("the cat with the dogs near a robot runs")
    assert not g.is_grammatical("the cat with the dogs near a robot run")

    # NP = "the robots near the senators with a cat" -> nearest noun = "cat" (sg) -> main verb must be sg
    assert g.is_grammatical("the robots near the senators with a cat runs")
    assert not g.is_grammatical("the robots near the senators with a cat run")


def test_rejects_pp_after_vp_B():
    v = GrammarVocab()
    g = GrammarB(v)

    # GrammarB keeps the same surface structure constraints as GrammarA:
    # PPs are inside NP only, not after VP.
    assert not g.is_grammatical("the cat sleeps with the dogs")
    assert not g.is_grammatical("the cats run near a dog")


def test_transitives_main_clause_B_if_supported():
    v = GrammarVocab()
    g = GrammarB(v)

    # No PP: nearest noun is head noun, so basic transitive agreement works
    assert g.is_grammatical("the cat sees the dogs")     # nearest noun "cat" sg -> "sees" sg
    assert g.is_grammatical("the cats see the dog")      # nearest noun "cats" pl -> "see" pl

    assert not g.is_grammatical("the cat see the dogs")  # sg nearest -> pl verb
    assert not g.is_grammatical("the cats sees the dog") # pl nearest -> sg verb

    # With PP: nearest noun is PP object noun
    # "the cat with the dogs see the robot" -> nearest noun before verb = "dogs" (pl) -> verb must be pl ("see")
    assert g.is_grammatical("the cat with the dogs see the robot")
    assert not g.is_grammatical("the cat with the dogs sees the robot")

    # TV needs object NP
    assert not g.is_grammatical("the cat with the dogs see")


def test_rejects_unknown_tokens_and_bad_starts_B():
    v = GrammarVocab()
    g = GrammarB(v)

    assert not g.is_grammatical("cat sleeps")               # missing det
    assert not g.is_grammatical("the green cat sleeps")     # "green" not in vocab
    assert not g.is_grammatical("the cat quickly sleeps")   # "quickly" not in vocab


if __name__ == "__main__":
    # Grammar A
    test_generator_outputs_are_grammatical_A()
    test_rejects_subject_verb_disagreement_simple_A()
    test_accepts_pp_inside_np_A()
    test_rejects_pp_after_vp_A()
    test_rejects_unknown_tokens_and_bad_starts_A()
    test_transitives_main_clause_A_if_supported()

    # Grammar B
    test_generator_outputs_are_grammatical_B()
    test_nearest_noun_agreement_signature_B()
    test_accepts_pp_inside_np_B_and_tracks_nearest_noun_B()
    test_rejects_pp_after_vp_B()
    test_transitives_main_clause_B_if_supported()
    test_rejects_unknown_tokens_and_bad_starts_B()

    print("All tests passed.")