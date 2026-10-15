"""
engines/forced_trade/signature_sizing.py

Cycle 66 T4 -- SIZE the path-signature question. No signature is computed
here and no feature is extracted: this module is arithmetic on term counts set
against the labelled sample size, plus the record of which signature packages
install on this estate's Python 3.12 (measured in throwaway venvs, never in
the project .venv).

Term counts for a d-channel path truncated at level N:
    signature      sum_{k=1..N} d^k                     (d=5, N=4 -> 780)
    log-signature  sum_{k=1..N} (1/k) sum_{j|k} mu(j) d^(k/j)   (Witt's formula,
                   the number of Lyndon words; d=5, N=4 -> 205)
"""
from __future__ import annotations

import pandas as pd

# Channel sets a cascade signature would plausibly use. Time augmentation is
# standard (it makes the signature sensitive to speed, not just shape), and
# lead-lag doubles the channel count to capture quadratic variation.
CHANNEL_SETS = (
    (4, "price, signed flow, depth, spread"),
    (5, "+ time augmentation"),
    (10, "lead-lag of the 5"),
)
LEVELS = (2, 3, 4)


def mobius(n: int) -> int:
    if n == 1:
        return 1
    res, m, p = 1, n, 2
    while p * p <= m:
        if m % p == 0:
            m //= p
            if m % p == 0:
                return 0
            res = -res
        p += 1
    if m > 1:
        res = -res
    return res


def sig_terms(d: int, n: int) -> int:
    return sum(d ** k for k in range(1, n + 1))


def logsig_terms(d: int, n: int) -> int:
    total = 0
    for k in range(1, n + 1):
        s = sum(mobius(j) * d ** (k // j) for j in range(1, k + 1) if k % j == 0)
        total += s // k
    return total


def sizing_table(samples: dict) -> pd.DataFrame:
    """Terms per (channels, level), and terms per labelled cluster for each
    candidate supervised sample. Ratio >= 10 is an order of magnitude more
    features than labels: the finding that makes the study premature."""
    rows = []
    for d, label in CHANNEL_SETS:
        for n in LEVELS:
            r = {"channels": d, "channel_set": label, "level": n,
                 "sig_terms": sig_terms(d, n), "logsig_terms": logsig_terms(d, n)}
            for name, m in samples.items():
                r[f"sig_per_label[{name}]"] = (r["sig_terms"] / m) if m else float("inf")
                r[f"logsig_per_label[{name}]"] = (r["logsig_terms"] / m) if m else float("inf")
            rows.append(r)
    return pd.DataFrame(rows)
