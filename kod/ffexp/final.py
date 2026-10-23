"""Confirmatory and exploratory analysis after the blind test evaluation (PROTOKOL_v5 §7).

Usage:  python -m ffexp.final --results WYNIKI --params PARAMETRY.json --out ANALIZA.json

PARAMETRY.json holds the frozen values derived from the mini-pilot (S per dataset and
hypothesis family, N).
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import torch

from . import analysis as A
from . import models

BOOT_REPS = 10000


def load_test(results, stage):
    """{(dataset, arm, w): {seed: record}} with per-example correctness arrays."""
    out = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(results, "test", stage, "*.pt"))):
        name = os.path.basename(path)[:-3]
        with open(os.path.join(results, stage if not stage.endswith("_pierwotne") else stage[:-10],
                               "runs" if not stage.endswith("_pierwotne") else "runs_pierwotne",
                               name + ".json")) as f:
            run = json.load(f)
        j = run["job"]
        res = torch.load(path, weights_only=False)
        nan = any(run["nan"].values())
        out[(j["dataset"], j["arm"], j.get("w", 250))][j["seed"]] = {
            k: {kk: vv.numpy().astype(np.float32) for kk, vv in v.items()} for k, v in res.items()
        } | {"_nan": nan}
    return out


def _acc(rec, which="correct", test_set=None, ds=None):
    return float(rec[test_set or ds][which].mean() * 100)


def diffs(table, ds, b_arm, group, test_set=None, which="correct"):
    seeds = sorted(table[(ds, b_arm, 250)])
    for k in group:
        assert sorted(table[(ds, k, 250)]) == seeds, f"unpaired seeds {ds} {k}"
    acc = lambda arm: np.array([_acc(table[(ds, arm, 250)][s], which, test_set, ds) for s in seeds])  # noqa: E731
    return np.stack([acc(b_arm) - acc(k) for k in group], 1), seeds


def any_nan(table, ds, arms):
    return any(table[(ds, a, 250)][s]["_nan"] for a in arms for s in table[(ds, a, 250)])


def bootstrap_means(table, ds, b_arm, group, test_set=None, reps=BOOT_REPS, seed=0):
    """§7.8(a): independent resampling of seeds and test examples. Returns (reps, m)."""
    rng = np.random.default_rng(seed)
    seeds = sorted(table[(ds, b_arm, 250)])
    key = test_set or ds
    mat = {a: np.stack([table[(ds, a, 250)][s][key]["correct"] for s in seeds]) for a in [b_arm] + group}
    n_seeds, n_ex = mat[b_arm].shape
    out = np.empty((reps, len(group)))
    chunk = 500
    for start in range(0, reps, chunk):
        r = min(chunk, reps - start)
        w = rng.multinomial(n_ex, np.full(n_ex, 1.0 / n_ex), size=r).T.astype(np.float32)  # (n_ex, r)
        s_idx = rng.integers(0, n_seeds, size=(r, n_seeds))
        acc = {a: (m @ w) / n_ex * 100 for a, m in mat.items()}  # (n_seeds, r)
        b = np.take_along_axis(acc[b_arm].T, s_idx, 1).mean(1)
        for j, k in enumerate(group):
            out[start : start + r, j] = b - np.take_along_axis(acc[k].T, s_idx, 1).mean(1)
    return out


def confirmatory(table, params, test_set_mnist=None):
    out = {}
    for h, (b_arm, group) in models.HYPOTHESES.items():
        fam = "S_H3" if h == "H3" else "S_H1H2"
        per_ds = {}
        for ds in ("mnist", "fmnist"):
            S = params["S"][ds][fam]
            ts = test_set_mnist if ds == "mnist" else None
            d, seeds = diffs(table, ds, b_arm, group, ts)
            v = A.verdict_one(d, S, "t")
            vw = A.verdict_one(d, S, "wilcoxon")
            v["wilcoxon"] = {"category": vw["category"]}
            v["nan_present"] = any_nan(table, ds, [b_arm] + group)
            if v["nan_present"] and vw["category"] != v["category"]:
                v["category_main"] = v["category"]
                v["category"], v["class"] = "NIEROZSTRZYGNIETY (niestabilnosc)", "N"
            v["n_seeds"] = len(seeds)
            v["S"] = S
            v["comparators"] = group
            per_ds[ds] = v
        out[h] = {"per_dataset": per_ds, "joint": A.combine(per_ds["mnist"], per_ds["fmnist"])}
    return out


def sensitivity_bootstrap(table, params):
    out = {}
    for h, (b_arm, group) in models.HYPOTHESES.items():
        fam = "S_H3" if h == "H3" else "S_H1H2"
        per = {ds: A.verdict_from_boot(bootstrap_means(table, ds, b_arm, group), params["S"][ds][fam])
               for ds in ("mnist", "fmnist")}
        out[h] = {"per_dataset": per, "joint": A.combine(per["mnist"], per["fmnist"])}
    return out


def sensitivity_qmnist(table, params):
    out = {}
    for h, (b_arm, group) in models.HYPOTHESES.items():
        fam = "S_H3" if h == "H3" else "S_H1H2"
        d, _ = diffs(table, "mnist", b_arm, group, test_set="qmnist")
        out[h] = A.verdict_one(d, params["S"]["mnist"][fam], "t")
    return out


def _mean_ci(d):
    d = np.asarray(d)
    from scipy import stats
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    return {"mean": float(d.mean()), "ci95": [float(d.mean() - q * se), float(d.mean() + q * se)], "n": n}


def exploratory(table, e2, e9):
    ex = {}
    accs = {}
    for (ds, arm, w), recs in table.items():
        seeds = sorted(recs)
        accs[(ds, arm)] = {s: _acc(recs[s], ds=ds) for s in seeds}
    pair = lambda ds, a, b: _mean_ci([accs[(ds, a)][s] - accs[(ds, b)][s] for s in sorted(accs[(ds, a)])])  # noqa
    ex["E1_depth"] = {ds: {x: pair(ds, "K-" + k, "P-" + p) for x, (k, p) in
                           {"R": ("R", "R"), "L": ("L", "L"), "G": ("G", "G"), "T": ("T", "T"),
                            "BANK": ("BANK", "BANK")}.items()} for ds in ("mnist", "fmnist")}
    ex["E3_single_contrasts"] = {
        ds: {f"{b}-{k}": pair(ds, b, k) for h, (b, g) in models.HYPOTHESES.items() for k in g}
        | {f"K-BANK-{k}": pair(ds, "K-BANK", k) for k in ["K-LGT"] + models.H1_GROUP}
        | {f"{p}-{k}": pair(ds, p, k) for p in models.H2_GROUP for k in models.H1_GROUP}
        for ds in ("mnist", "fmnist")}
    ex["E6_g2_only"] = {ds: {arm: _mean_ci([_acc(table[(ds, arm, 250)][s], "correct_g2only", ds=ds)
                                            for s in sorted(table[(ds, arm, 250)])]) for arm in models.FF_ARMS}
                        for ds in ("mnist", "fmnist")}
    e7 = {}
    for ds in ("mnist", "fmnist"):
        seeds = sorted(accs[(ds, "K-LGT")])
        tanh_out = np.mean([[accs[(ds, a)][s] for a in ("K-LGT", "K-GLT")] for s in seeds], 1)
        other = np.mean([[accs[(ds, a)][s] for a in ("K-LTG", "K-GTL", "K-TLG", "K-TGL")] for s in seeds], 1)
        e7[ds] = {"tanh_out_minus_other": _mean_ci(tanh_out - other), "LGT-GLT": pair(ds, "K-LGT", "K-GLT"),
                  **{f"LGT-{k}": pair(ds, "K-LGT", k) for k in ("K-T", "K-LLT", "K-GGT")}}
    ex["E7_tanh_output"] = e7
    if e2:
        ex["E2_bp_gap"] = {}
        for ds in ("mnist", "fmnist"):
            res = {}
            for bp, ff in models.BP_ARMS.items():
                seeds = sorted(e2[(ds, bp, 250)])
                res[f"{bp}-{ff}"] = _mean_ci([_acc(e2[(ds, bp, 250)][s], ds=ds) - accs[(ds, ff)][s] for s in seeds])
            ex["E2_bp_gap"][ds] = res
    if e9:
        curves = {}
        for ds in ("mnist", "fmnist"):
            for arm in models.E9_ARMS:
                pts = {}
                for w in models.E9_WIDTHS:
                    recs = e9[(ds, arm, w)]
                    pts[w] = _mean_ci([_acc(recs[s], ds=ds) for s in sorted(recs)])
                seeds10 = sorted(e9[(ds, arm, models.E9_WIDTHS[0])])
                pts[250] = _mean_ci([accs[(ds, arm)][s] for s in seeds10])
                curves[f"{ds}|{arm}"] = pts
        ex["E9_width_curves"] = curves
    return ex


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.params) as f:
        params = json.load(f)
    table = load_test(a.results, "C")
    e2 = load_test(a.results, "E2")
    e9 = load_test(a.results, "E9")
    report = {
        "confirmatory": confirmatory(table, params),
        "sensitivity_a_bootstrap": sensitivity_bootstrap(table, params),
        "sensitivity_d_qmnist": sensitivity_qmnist(table, params),
        "exploratory": exploratory(table, e2, e9),
    }
    originals = load_test(a.results, "C_pierwotne")
    if originals:
        substituted = {k: dict(v) for k, v in table.items()}
        for k, recs in originals.items():
            substituted[k].update(recs)
        report["sensitivity_c_originals"] = confirmatory(substituted, params)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    for h, v in report["confirmatory"].items():
        print(h, v["joint"]["category"], v["joint"]["per_dataset"])


if __name__ == "__main__":
    main()
