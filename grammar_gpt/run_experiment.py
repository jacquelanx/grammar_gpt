from __future__ import annotations

import argparse
import os
import json
from dataclasses import asdict

import torch

from .tokenization import SimpleTokenizer
from .grammars import GrammarVocab
from .data_pipeline import DataSpec, build_datasets
from .train import TrainConfig, train_one
from .gpt_small import GPTConfig, GPTSmall


def load_tokenizer(path: str) -> SimpleTokenizer:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return SimpleTokenizer(vocab=obj["vocab"], mode=obj["mode"], unk_token=obj["unk_token"], eos_token=obj["eos_token"])


def load_model_for_eval(ckpt_path: str, device: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    gcfg = GPTConfig(**ckpt["gpt_config"])
    model = GPTSmall(gcfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def run_eval_inline(ckpt: str, out_dir: str, device: str):
    from .suites import EvalSpec, build_conflict_suite, build_sanity_suite, build_holdout_lexrole_suite
    from .scoring import score_pair
    from .stats import summarize_pair_results, groupby
    from .typology import error_typology_pair
    from .generation import GenEvalConfig, run_generation_suite
    from .grammars import GrammarA, GrammarB
    from .probes import AttnProbeSpec, ProbeSpec, attention_probe_subject_vs_intervener, run_probe_subject_number

    os.makedirs(out_dir, exist_ok=True)

    vocab = GrammarVocab()
    tok = load_tokenizer(os.path.join(os.path.dirname(out_dir), "..", "data", "tokenizer.json")) if False else SimpleTokenizer(vocab=vocab.all_word_tokens(), mode="word")

    model = load_model_for_eval(ckpt, device=device)

    spec = EvalSpec(seed=123, n_conflict=2000, n_sanity=1000, n_holdout_lexrole=1000)
    suite_conf = build_conflict_suite(vocab, spec)
    suite_san = build_sanity_suite(vocab, spec)
    suite_hold = build_holdout_lexrole_suite(vocab, spec)

    res_conf = [score_pair(model, tok, ex, device) for ex in suite_conf]
    res_san = [score_pair(model, tok, ex, device) for ex in suite_san]
    res_hold = [score_pair(model, tok, ex, device) for ex in suite_hold]

    summary = {
        "suite_conflict": summarize_pair_results(res_conf),
        "suite_sanity": summarize_pair_results(res_san),
        "suite_holdout_lexrole": summarize_pair_results(res_hold),
    }

    gb = groupby(res_conf, key_fn=lambda r: (r.meta.get("subject_num"), r.meta.get("intervener_num"), r.meta.get("p")))
    summary["suite_conflict_breakdowns"] = {k: summarize_pair_results(v) for k, v in gb.items()}

    counts = {}
    for r in res_conf:
        t = error_typology_pair(r, vocab)
        counts[t] = counts.get(t, 0) + 1
    summary["suite_conflict_error_typology_counts"] = counts

    grammarA = GrammarA(vocab)
    grammarB = GrammarB(vocab)
    summary["suite_generation"] = run_generation_suite(model, tok, device, GenEvalConfig(), grammarA, grammarB)

    try:
        summary["attention_probe"] = attention_probe_subject_vs_intervener(model, tok, device, vocab, AttnProbeSpec())
    except Exception as e:
        summary["attention_probe"] = {"error": str(e)}

    try:
        summary["probe_subject_number"] = run_probe_subject_number(model, tok, device, grammarA, ProbeSpec())
    except Exception as e:
        summary["probe_subject_number"] = {"error": str(e)}

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_out", type=str, default="runs")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--make_data", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.root_out, exist_ok=True)

    # 1) Data
    data_dir = os.path.join(args.root_out, "data")
    if args.make_data or not os.path.exists(data_dir):
        spec = DataSpec(seed=args.seed, out_dir=data_dir)
        paths = build_datasets(spec)
    else:
        paths = {
            "tokenizer": os.path.join(data_dir, "tokenizer.json"),
            "A_train": os.path.join(data_dir, "A_train_blocks.pt"),
            "A_val": os.path.join(data_dir, "A_val_blocks.pt"),
            "B_train": os.path.join(data_dir, "B_train_blocks.pt"),
            "B_val": os.path.join(data_dir, "B_val_blocks.pt"),
            "C_train": os.path.join(data_dir, "C_train_blocks.pt"),
            "C_val": os.path.join(data_dir, "C_val_blocks.pt"),
        }

    vocab = GrammarVocab()
    tok = SimpleTokenizer(vocab=vocab.all_word_tokens(), mode="word")
    vocab_size = tok.vocab_size

    # 2) Train A/B/C
    for cond in ["A", "B", "C"]:
        out_dir = os.path.join(args.root_out, f"train_{cond}")
        os.makedirs(out_dir, exist_ok=True)

        tcfg = TrainConfig(
            seed=args.seed,
            device=args.device,
            train_blocks_path=paths[f"{cond}_train"],
            val_blocks_path=paths[f"{cond}_val"],
            block_size=64,
            vocab_size=vocab_size,
            out_dir=out_dir,
            # sane defaults for laptop; adjust max_steps to your token budget
            batch_size=64,
            max_steps=8000,
            lr=3e-4,
            warmup_steps=200,
            n_layer=6,
            n_head=6,
            n_embd=384,
            dropout=0.1,
        )

        best_ckpt = train_one(tcfg)

        # 3) Eval
        eval_out = os.path.join(args.root_out, f"eval_{cond}")
        run_eval_inline(best_ckpt, eval_out, args.device)


if __name__ == "__main__":
    main()