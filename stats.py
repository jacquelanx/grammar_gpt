from __future__ import annotations

from typing import Callable, Dict, List, Tuple
import random
import statistics

from .scoring import PairResult


# Confidence interval
def bootstrap_ci( 
    values: List[float],
    stat_fn: Callable[[List[float]], float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 123,
) -> Tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(values)
    point = stat_fn(values)
    boots: List[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(stat_fn(sample))
    boots.sort()
    alpha = (1.0 - ci) / 2.0
    lo = boots[int(alpha * n_boot)]
    hi = boots[int((1.0 - alpha) * n_boot) - 1]
    return point, lo, hi


# Delta statistics
def summarize_pair_results(results: List[PairResult], eps_uncertain: float = 1e-4) -> Dict[str, float | List[float] | int]:
    accs = [float(r.correct) for r in results]
    deltas = [float(r.delta) for r in results]
    uncertain = [1.0 if abs(d) < eps_uncertain else 0.0 for d in deltas]

    acc_point, acc_lo, acc_hi = bootstrap_ci(accs, lambda xs: sum(xs) / len(xs))
    d_point, d_lo, d_hi = bootstrap_ci(deltas, lambda xs: sum(xs) / len(xs))

    return {
        "n": len(results),
        "acc": acc_point,
        "acc_ci95": [acc_lo, acc_hi],
        "mean_delta": d_point,
        "mean_delta_ci95": [d_lo, d_hi],
        "median_delta": float(statistics.median(deltas)) if deltas else float("nan"),
        "uncertain_frac": float(sum(uncertain) / max(len(uncertain), 1)),
    }


def groupby(results: List[PairResult], key_fn) -> Dict[str, List[PairResult]]:
    out: Dict[str, List[PairResult]] = {}
    for r in results:
        k = str(key_fn(r))
        out.setdefault(k, []).append(r)
    return out