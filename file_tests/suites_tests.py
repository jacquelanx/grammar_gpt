from grammars import GrammarVocab
from suites import (
    EvalSpec,
    build_conflict_suite,
    build_sanity_suite,
    build_holdout_lexrole_suite,
)


def print_examples(name, examples, k=5):
    print("=" * 60)
    print(f"{name} (showing {k} examples)")
    print("=" * 60)
    for i, ex in enumerate(examples[:k]):
        print(f"[{i}] GRAM:   {ex.grammatical}")
        print(f"    UNGRAM: {ex.ungrammatical}")
        print(f"    META:   {ex.meta}")
        print()


def main():
    vocab = GrammarVocab()
    spec = EvalSpec(
        n_conflict=10,
        n_sanity=10,
        n_holdout_lexrole=10,
    )

    conflict = build_conflict_suite(vocab, spec)
    sanity = build_sanity_suite(vocab, spec)
    holdout = build_holdout_lexrole_suite(vocab, spec)

    print_examples("CONFLICT SUITE", conflict)
    print_examples("SANITY SUITE", sanity)
    print_examples("HOLDOUT LEXICAL ROLE SUITE", holdout)


if __name__ == "__main__":
    main()
