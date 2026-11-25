"""Statistics for paired accuracy comparisons."""
from __future__ import annotations

import random

from scipy.stats import binomtest


def bootstrap_ci(xs: list[bool | float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    if not xs:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(xs)
    mean = sum(xs) / n
    boots = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return mean, boots[int(alpha / 2 * n_boot)], boots[int((1 - alpha / 2) * n_boot) - 1]


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact McNemar test on paired binary outcomes (a vs b on the same items)."""
    assert len(a) == len(b)
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = a_only + b_only
    p = 1.0 if n == 0 else binomtest(a_only, n, 0.5).pvalue
    return {"a_only": a_only, "b_only": b_only, "p_value": p}
