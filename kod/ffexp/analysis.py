"""Decision rules, power simulation and mini-pilot derivation (PROTOKOL_v5 §7, §9.2, §9.3).

All elementary conditions of §7.3 are evaluated through one-sided p-values, which
is equivalent to the confidence-interval formulation for the t-test and lets the
same code serve the Wilcoxon (§7.6) and bootstrap (§7.8a) variants:

  sup(D)  <=> for every K: p(H0: Δ_K <= D) < α'/2      (lower bound of the (1-α') CI > D)
  eq      <=> for every K: max(p(Δ_K <= -S), p(Δ_K >= S)) < α'   (TOST, (1-2α') CI)
  inf(D)  <=> Holm over K of p(H0: Δ_K >= -D) at level α'/2 rejects >= 1,
              i.e. min_K p <= (α'/2) / m
"""
import math

import numpy as np
from scipy import stats

ALPHA = 0.05
ALPHA1 = ALPHA / 3  # α' (Bonferroni over H1-H3)

CATEGORIES = ["GORSZOSC", "PRZEWAGA", "ROWNOWAZNOSC+", "GORSZOSC<S", "PRZEWAGA<S", "ROWNOWAZNOSC",
              "NIEROZSTRZYGNIETY"]
CLASS_OF = {"GORSZOSC": "G", "PRZEWAGA": "P", "ROWNOWAZNOSC+": "R", "GORSZOSC<S": "G", "PRZEWAGA<S": "P",
            "ROWNOWAZNOSC": "R", "NIEROZSTRZYGNIETY": "N"}
STRENGTH = {"PRZEWAGA": 2, "PRZEWAGA<S": 1, "GORSZOSC": 2, "GORSZOSC<S": 1, "ROWNOWAZNOSC+": 2,
            "ROWNOWAZNOSC": 1}


# ---------------------------------------------------------------- p-values
def _t_pvals(d, delta):
    """One-sided t p-values for columns of d (..., N, m). Returns (p_greater, p_less) for H1: mean > / < delta."""
    n = d.shape[-2]
    mean = d.mean(axis=-2)
    sd = d.std(axis=-2, ddof=1)
    se = np.where(sd > 0, sd / math.sqrt(n), np.nan)
    t = (mean - delta) / se
    # sd == 0: degenerate; decide by sign of (mean - delta)
    pg = np.where(np.isnan(t), np.where(mean > delta, 0.0, 1.0), stats.t.sf(np.nan_to_num(t), n - 1))
    pl = np.where(np.isnan(t), np.where(mean < delta, 0.0, 1.0), stats.t.cdf(np.nan_to_num(t), n - 1))
    return pg, pl


def _wilcoxon_pvals(d, delta):
    m = d.shape[-1]
    pg, pl = np.ones(m), np.ones(m)
    for k in range(m):
        x = d[:, k] - delta
        if np.allclose(x, 0):
            continue
        pg[k] = stats.wilcoxon(x, alternative="greater", zero_method="wilcox").pvalue
        pl[k] = stats.wilcoxon(x, alternative="less", zero_method="wilcox").pvalue
    return pg, pl


def pvals(d, delta, method="t"):
    if method == "t":
        return _t_pvals(d, delta)
    if method == "wilcoxon":
        return _wilcoxon_pvals(d, delta)
    raise ValueError(method)


# ---------------------------------------------------------------- §7.3
def categorize_from_p(p_sup_S, p_sup_0, p_eq_lo, p_eq_hi, p_inf_S, p_inf_0, alpha1=ALPHA1):
    """Vectorized category assignment. Inputs have shape (..., m). Returns int array of category index."""
    m = p_sup_S.shape[-1]
    sup_S = np.all(p_sup_S < alpha1 / 2, axis=-1)
    sup_0 = np.all(p_sup_0 < alpha1 / 2, axis=-1)
    eq = np.all(np.maximum(p_eq_lo, p_eq_hi) < alpha1, axis=-1)
    inf_S = np.min(p_inf_S, axis=-1) <= (alpha1 / 2) / m
    inf_0 = np.min(p_inf_0, axis=-1) <= (alpha1 / 2) / m
    cat = np.full(sup_S.shape, 6)
    for idx, cond in reversed(list(enumerate([inf_S, sup_S, eq & (sup_0 | inf_0), inf_0, sup_0, eq]))):
        cat = np.where(cond, idx, cat)
    direction = np.where(sup_0, 1, np.where(inf_0, -1, 0))
    return cat, direction


def verdict_one(d, S, method="t", alpha1=ALPHA1):
    """d: (N, m) differences B - K_j. Returns dict with category and details."""
    d = np.asarray(d, dtype=float)
    pg_S, _ = pvals(d, S, method)          # sup(S): H1 mean > S
    pg_0, pl_0 = pvals(d, 0.0, method)     # sup(0) / inf(0)
    pg_mS, _ = pvals(d, -S, method)        # eq lower: H1 mean > -S
    _, pl_S = pvals(d, S, method)          # eq upper: H1 mean < S
    _, pl_mS = pvals(d, -S, method)        # inf(S): H1 mean < -S
    cat, direction = categorize_from_p(pg_S, pg_0, pg_mS, pl_S, pl_mS, pl_0, alpha1)
    n = d.shape[0]
    mean, sd = d.mean(0), d.std(0, ddof=1)
    tq = stats.t.ppf(1 - alpha1 / 2, n - 1)
    return {
        "category": CATEGORIES[int(cat)], "class": CLASS_OF[CATEGORIES[int(cat)]], "direction": int(direction),
        "eq": bool(np.all(np.maximum(pg_mS, pl_S) < alpha1)),
        "mean": mean.tolist(), "sd": sd.tolist(),
        "ci": np.stack([mean - tq * sd / math.sqrt(n), mean + tq * sd / math.sqrt(n)], 1).tolist(),
    }


def verdict_from_boot(boot_means, S, alpha1=ALPHA1):
    """§7.8(a): boot_means (R, m) bootstrap replicates of the mean differences."""
    b = np.asarray(boot_means)
    m = b.shape[1]
    p = lambda cond: cond.mean(axis=0)  # noqa: E731
    cat, direction = categorize_from_p(
        p(b <= S), p(b <= 0), p(b <= -S), p(b >= S), p(b >= -S), p(b >= 0), alpha1)
    ci = np.quantile(b, [alpha1 / 2, 1 - alpha1 / 2], axis=0).T
    return {"category": CATEGORIES[int(cat)], "class": CLASS_OF[CATEGORIES[int(cat)]], "direction": int(direction),
            "ci": ci.tolist(), "m": m}


# ---------------------------------------------------------------- §7.4
def combine(v1, v2):
    """Combine per-dataset verdicts (dicts with category/class/direction) into the joint verdict."""
    c1, c2 = v1["class"], v2["class"]
    k1, k2 = v1["category"], v2["category"]
    if c1 == c2 and c1 != "N":
        weaker = k1 if STRENGTH[k1] <= STRENGTH[k2] else k2
        if c1 == "R":
            weaker = "ROWNOWAZNOSC+" if "ROWNOWAZNOSC+" in (k1, k2) else "ROWNOWAZNOSC"
        return {"category": weaker, "class": c1, "per_dataset": [k1, k2]}
    if c1 == "N" and c2 == "N":
        return {"category": "NIEROZSTRZYGNIETY", "class": "N", "per_dataset": [k1, k2]}
    pair = {c1, c2}
    if pair == {"P", "G"}:
        cat = "NIESPOJNY-KIERUNEK"
    else:
        cat = "NIESPOJNY-CZESCIOWY"
        for a, b in ((v1, v2), (v2, v1)):
            if a["class"] == "P" and b["category"] == "ROWNOWAZNOSC+" and b["direction"] > 0:
                cat = "NIESPOJNY-ZGODNY-KIERUNEK"
            if a["class"] == "G" and b["category"] == "ROWNOWAZNOSC+" and b["direction"] < 0:
                cat = "NIESPOJNY-ZGODNY-KIERUNEK"
    return {"category": cat, "class": "N", "per_dataset": [k1, k2]}


def _combine_class_vec(cat1, dir1, cat2, dir2):
    """Vectorized joint class (P/R/G/N as 0/1/2/3) for simulations."""
    cls = np.array([2, 0, 1, 2, 0, 1, 3])  # category index -> class index (P=0, R=1, G=2, N=3)
    k1, k2 = cls[cat1], cls[cat2]
    return np.where((k1 == k2) & (k1 != 3), k1, 3)


# ---------------------------------------------------------------- power (§7.5, §9.2)
def simulate(n, m, means, sigma, rho, S, reps=20000, rng=None, method="t"):
    """Simulate per-dataset categories. means: (m,) true Δ. Returns (cat, dir) arrays of length reps."""
    rng = np.random.default_rng(rng)
    rho = max(0.0, min(rho, 0.999))
    cov = sigma ** 2 * ((1 - rho) * np.eye(m) + rho * np.ones((m, m)))
    L = np.linalg.cholesky(cov)
    z = rng.standard_normal((reps, n, m)) @ L.T + np.asarray(means)
    pg_S, _ = _t_pvals(z, S)
    pg_0, pl_0 = _t_pvals(z, 0.0)
    pg_mS, pl_mS = _t_pvals(z, -S)
    _, pl_S = _t_pvals(z, S)
    return categorize_from_p(pg_S, pg_0, pg_mS, pl_S, pl_mS, pl_0)


def joint_rates(n, m, means_by_ds, sigma_by_ds, rho, S_by_ds, reps=20000, seed=0):
    """Joint (both datasets) class rates and per-dataset PRZEWAGA rate."""
    res = [simulate(n, m, means_by_ds[i], sigma_by_ds[i], rho, S_by_ds[i], reps, seed + i) for i in range(2)]
    joint = _combine_class_vec(res[0][0], res[0][1], res[1][0], res[1][1])
    out = {c: float(np.mean(joint == i)) for i, c in enumerate("PRGN")}
    out["PRZEWAGA_joint"] = float(np.mean((res[0][0] == 1) & (res[1][0] == 1)))
    return out


def design_powers(n, m, S, sigma, rho, reps=20000):
    """Power targets of §7.5 for both datasets. S, sigma: length-2 sequences."""
    sc_i = [np.array([S[i]] + [2 * S[i]] * (m - 1)) for i in range(2)]
    sc_ii = [np.zeros(m) for _ in range(2)]
    sc_big = [np.array([2 * S[i]] + [3 * S[i]] * (m - 1)) for i in range(2)]
    p_i = joint_rates(n, m, sc_i, sigma, rho, S, reps, 1)["P"]
    p_ii = joint_rates(n, m, sc_ii, sigma, rho, S, reps, 2)["R"]
    p_big = joint_rates(n, m, sc_big, sigma, rho, S, reps, 3)["PRZEWAGA_joint"]
    return {"power_classP": p_i, "power_classR": p_ii, "power_PRZEWAGA_2S3S": p_big}


def choose_n(S, sigma, rho, m=6, candidates=(30, 40, 50, 60), reps=20000):
    table = {}
    for n in candidates:
        table[n] = design_powers(n, m, S, sigma, rho, reps)
        if table[n]["power_classP"] >= 0.80 and table[n]["power_classR"] >= 0.80:
            return n, table
    return candidates[-1], table


def round_up(x, step=0.05):
    return math.ceil(x / step - 1e-9) * step


def derive_minipilot(runs, val_preds=None, sigma_factor=1.293, reps=20000):
    """§9.3 / §7.5 / §5.2: derive S, sigma_d, rho, E_max and N from mini-pilot run records.

    runs: list of run JSON records (stage MP). val_preds: {run id: (val_idx, correct bool array)}.
    """
    from collections import defaultdict

    acc = defaultdict(dict)
    best_epochs = []
    for r in runs:
        j = r["job"]
        acc[(j["dataset"], j["arm"])][j["seed"]] = r["val_acc"]
        best_epochs += [e for e in r["best_epoch"].values() if e is not None]
    out = {"datasets": {}}
    contrasts = [("K-LGT", "K-G"), ("K-LGT", "K-T"), ("K-LGT", "K-GGT"), ("K-LGT", "K-GLT"),
                 ("P-BANK", "P-R"), ("P-BANK", "P-T")]
    S_list, sig_list, sig147, rhos = [], [], [], []
    for ds in ("mnist", "fmnist"):
        seeds = sorted(acc[(ds, "K-LGT")])
        a = lambda arm: np.array([acc[(ds, arm)][s] for s in seeds])  # noqa: E731
        err_k, err_p = 100 - a("K-LGT").mean(), 100 - a("P-R").mean()
        S_k, S_p = round_up(0.10 * err_k), round_up(0.10 * err_p)
        sds = {f"{x}-{y}": float(np.std(a(x) - a(y), ddof=1)) for x, y in contrasts}
        diffs_k = np.stack([a("K-LGT") - a(y) for y in ("K-G", "K-T", "K-GGT", "K-GLT")], 1)
        corr = np.corrcoef(diffs_k.T)
        rho_hat = float(np.mean(corr[np.triu_indices(4, 1)]))
        S_min = min(S_k, S_p)
        max_sd = max(sds.values())
        sigma = max(sigma_factor * max_sd, 0.5 * S_min)
        S_list.append(S_min), sig_list.append(sigma), rhos.append(rho_hat)
        sig147.append(max(1.469 * max_sd, 0.5 * S_min))
        info = {"err_K-LGT": err_k, "err_P-R": err_p, "S_H1H2": S_k, "S_H3": S_p, "S_power": S_min,
                "sd_contrasts": sds, "max_sd": max_sd, "sigma_d": sigma, "rho_hat": rho_hat}
        if val_preds:
            info["example_sampling_se_pp"] = example_sampling_se(runs, val_preds, ds, contrasts)
        out["datasets"][ds] = info
    rho = float(min(max(np.mean(rhos), 0.0), 0.5))
    p90 = float(np.percentile(best_epochs, 90)) if best_epochs else 0.0
    out["emax"] = int(min(80, max(30, math.ceil(1.5 * p90))))
    out["best_epoch_p90"] = p90
    out["rho"] = rho
    n, table = choose_n(S_list, sig_list, rho, m=6, reps=reps)
    out["N"] = n
    out["power_table"] = {str(k): v for k, v in table.items()}
    out["power_at_sigma_1.47"] = design_powers(n, 6, S_list, sig147, rho, reps)
    return out


def example_sampling_se(runs, val_preds, ds, contrasts):
    """SE (pp) of the test-example sampling component, estimated on validation predictions.

    Per contrast: per-example difference in correctness averaged over seeds (examples are matched
    by global index; each seed has its own validation split, so only examples present in a split
    contribute to that seed's average)."""
    from collections import defaultdict

    corr = defaultdict(dict)
    for r in runs:
        j = r["job"]
        if j["dataset"] != ds or r["id"] not in val_preds:
            continue
        idx, ok = val_preds[r["id"]]
        corr[j["arm"]][j["seed"]] = (np.asarray(idx), np.asarray(ok, dtype=float))
    out = {}
    for x, y in contrasts:
        tot, cnt = np.zeros(60000), np.zeros(60000)
        for s in corr[x]:
            if s not in corr[y]:
                continue
            (ix, ox), (iy, oy) = corr[x][s], corr[y][s]
            assert np.array_equal(ix, iy)
            tot[ix] += ox - oy
            cnt[ix] += 1
        mask = cnt > 0
        per_ex = tot[mask] / cnt[mask]
        out[f"{x}-{y}"] = float(per_ex.std(ddof=1) / math.sqrt(10000) * 100)
    return out
