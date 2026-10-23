"""Pipeline simulation test of the analysis (PROTOKOL_v5 §9.2).

Usage:  python -m ffexp.selftest_analysis --n N --S-mnist S1 --S-fmnist S2 --sigma-mnist s1 --sigma-fmnist s2 --rho r

Checks (a), (a'), (b) type-I rates, (c), (d) powers, and (e) agreement of the vectorized
decision code with an independent loop-based reference written directly from the
confidence-interval definitions of §7.3, on 100 constructed cases (incl. NaN runs).
"""
import argparse
import json
import math

import numpy as np
from scipy import stats

from . import analysis as A

M = 6  # worst case (H1)


# ------------------------------------------------------------- reference (§7.3 literally)
def reference_category(d, S, a1=A.ALPHA1):
    n, m = d.shape
    mean, sd = d.mean(0), d.std(0, ddof=1)
    se = sd / math.sqrt(n)
    t1 = stats.t.ppf(1 - a1 / 2, n - 1)
    t2 = stats.t.ppf(1 - a1, n - 1)
    lo1, hi1 = mean - t1 * se, mean + t1 * se
    lo2, hi2 = mean - t2 * se, mean + t2 * se
    sup_S = all(lo1[k] > S for k in range(m))
    sup_0 = all(lo1[k] > 0 for k in range(m))
    eq = all(lo2[k] > -S and hi2[k] < S for k in range(m))

    def holm_any(delta):
        ps = sorted(stats.t.cdf((mean[k] + delta) / se[k], n - 1) for k in range(m))
        level = a1 / 2
        rejected = 0
        for i, p in enumerate(ps):
            if p <= level / (m - i):
                rejected += 1
            else:
                break
        return rejected > 0

    inf_S, inf_0 = holm_any(S), holm_any(0.0)
    if inf_S:
        return "GORSZOSC"
    if sup_S:
        return "PRZEWAGA"
    if eq and (sup_0 or inf_0):
        return "ROWNOWAZNOSC+"
    if inf_0:
        return "GORSZOSC<S"
    if sup_0:
        return "PRZEWAGA<S"
    if eq:
        return "ROWNOWAZNOSC"
    return "NIEROZSTRZYGNIETY"


def constructed_cases(rng, n):
    """100 cases spanning all categories, with occasional NaN-fallback runs (Δ ≈ -87 pp)."""
    cases = []
    scen = [
        ("all_big_plus", lambda S: np.full(M, 3 * S)),
        ("small_plus", lambda S: np.full(M, 0.4 * S)),
        ("zero", lambda S: np.zeros(M)),
        ("one_minus_big", lambda S: np.r_[-3 * S, np.full(M - 1, 2 * S)]),
        ("one_minus_small", lambda S: np.r_[-0.5 * S, np.zeros(M - 1)]),
        ("mixed", lambda S: np.r_[2 * S, 0.0, np.full(M - 2, 2 * S)]),
        ("edge_S", lambda S: np.full(M, S)),
    ]
    for i in range(100):
        name, fn = scen[i % len(scen)]
        S = [0.3, 1.2][i % 2]
        sigma = rng.choice([0.05, 0.2, 0.5]) * (1 if S < 1 else 3)
        d = fn(S) + rng.normal(0, sigma, (n, M))
        if i % 10 == 9:
            d[rng.integers(n), rng.integers(M)] = -87.0
        cases.append((name, S, d))
    return cases


def check_e(n, rng):
    mism = []
    verdicts = []
    for name, S, d in constructed_cases(rng, n):
        got = A.verdict_one(d, S)["category"]
        ref = reference_category(d, S)
        verdicts.append((name, got))
        if got != ref:
            mism.append((name, S, got, ref))
    # combination rule (§7.4), exhaustively over category pairs and directions
    comb_errors = []
    for k1 in A.CATEGORIES:
        for k2 in A.CATEGORIES:
            for d1 in (-1, 0, 1):
                for d2 in (-1, 0, 1):
                    v1 = {"category": k1, "class": A.CLASS_OF[k1], "direction": d1}
                    v2 = {"category": k2, "class": A.CLASS_OF[k2], "direction": d2}
                    got = A.combine(v1, v2)
                    exp_class = A.CLASS_OF[k1] if A.CLASS_OF[k1] == A.CLASS_OF[k2] != "N" else "N"
                    if got["class"] != exp_class:
                        comb_errors.append((k1, k2, d1, d2, got))
                    if {A.CLASS_OF[k1], A.CLASS_OF[k2]} == {"P", "G"} and got["category"] != "NIESPOJNY-KIERUNEK":
                        comb_errors.append((k1, k2, d1, d2, got))
    cats = sorted({c for _, c in verdicts})
    return {"mismatches": mism, "combine_errors": comb_errors, "categories_covered": cats}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--S-mnist", type=float, required=True)
    ap.add_argument("--S-fmnist", type=float, required=True)
    ap.add_argument("--sigma-mnist", type=float, required=True)
    ap.add_argument("--sigma-fmnist", type=float, required=True)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    S = [a.S_mnist, a.S_fmnist]
    sig = [a.sigma_mnist, a.sigma_fmnist]
    n, reps = a.n, a.reps
    mc = lambda p: 3 * math.sqrt(p * (1 - p) / reps)  # noqa: E731  Monte Carlo margin (3 SE)
    res = {}
    cls = np.array([2, 0, 1, 2, 0, 1, 3])
    for i, ds in enumerate(("mnist", "fmnist")):
        cat, _ = A.simulate(n, M, [0] + [2 * S[i]] * (M - 1), sig[i], a.rho, S[i], reps, 10 + i)
        r_a = float(np.mean(cls[cat] == 0))
        cat, _ = A.simulate(n, M, [S[i]] + [3 * S[i]] * (M - 1), sig[i], a.rho, S[i], reps, 20 + i)
        r_a2 = float(np.mean(cat == 1))
        cat, _ = A.simulate(n, M, np.zeros(M), sig[i], a.rho, S[i], reps, 30 + i)
        r_b = float(np.mean(cls[cat] == 2))
        lim = A.ALPHA1 / 2
        res[ds] = {"a_false_P": r_a, "a2_false_PRZEWAGA": r_a2, "b_false_G": r_b, "limit": lim,
                   "pass": all(r <= lim + mc(lim) for r in (r_a, r_a2, r_b))}
    pw = A.design_powers(n, M, S, sig, a.rho, reps * 5)
    res["c_d_powers"] = pw
    res["e"] = check_e(n, np.random.default_rng(0))
    res["pass"] = (res["mnist"]["pass"] and res["fmnist"]["pass"] and not res["e"]["mismatches"]
                   and not res["e"]["combine_errors"])
    print(json.dumps({k: v for k, v in res.items() if k != "e"}, indent=1))
    print("(e) niezgodności:", len(res["e"]["mismatches"]), "błędy łączenia:", len(res["e"]["combine_errors"]),
          "pokryte kategorie:", res["e"]["categories_covered"])
    print("WYNIK:", "ZALICZONY" if res["pass"] else "NIEZALICZONY")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
