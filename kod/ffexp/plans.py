"""Stage plans and data-dependent selection rules (PROTOKOL_v5 §5.2–5.5, §9.3).

CLI:
  python -m ffexp.plans minipilot OUT.json
  python -m ffexp.plans A OUT.json --emax E
  python -m ffexp.plans B OUT.json --results WYNIKI --emax E           (top-3 x 3 seeds)
  python -m ffexp.plans B2 OUT.json --results WYNIKI --emax E          (boundary rule)
  python -m ffexp.plans select-B WYBOR.json --results WYNIKI --emax E  (ETAP_B_WYBOR.json)
  python -m ffexp.plans emax-rule --results WYNIKI --stage A --emax E
  python -m ffexp.plans C OUT.json --choice ETAP_B_WYBOR.json --n N
"""
import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

from . import models

DATASETS = ["mnist", "fmnist"]
MINIPILOT_SEEDS = list(range(40001, 40011))   # mini-pilot 1 (technical, D4)
MINIPILOT2_SEEDS = list(range(40011, 40021))  # mini-pilot 2.0: the only source of derivations (D4)
STAGE_A_SEED = 30001
STAGE_B_SEEDS = [20001, 20002, 20003]
STAGE_C_SEED0 = 10000
E_SEEDS = list(range(10001, 10011))
MINIPILOT_CFG = {"lr": 1e-3, "wd": 1e-4, "c": 0.25}
MINIPILOT_EMAX = 60

# Search space (§5.3), log10 bounds.
AXES = {"lr": (-4.0, -2.0), "wd": (-6.0, -2.0), "c": (-2.0, 0.0)}
AXIS_ORDER = ["lr", "wd", "c"]


def sobol_points(m=16):
    from scipy.stats import qmc

    u = qmc.Sobol(d=3, scramble=True, seed=0).random(m)
    pts = []
    for row in u:
        pts.append({ax: float(10 ** (lo + (hi - lo) * v)) for (ax, (lo, hi)), v in zip(AXES.items(), row)})
    return pts


def _ff_job(stage, ds, arm, seed, cfg, emax, w=250, point=None):
    job = {"stage": stage, "kind": "ff", "dataset": ds, "arm": arm, "seed": seed, "cfg": cfg, "emax": emax, "w": w}
    if point is not None:
        job["point"] = point
    return job


def _bp_job(stage, ds, arm, seed, cfg, emax, point=None):
    job = {"stage": stage, "kind": "bp", "dataset": ds, "arm": arm, "seed": seed,
           "cfg": {"lr": cfg["lr"], "wd": cfg["wd"]}, "emax": emax, "w": 250}
    if point is not None:
        job["point"] = point
    return job


def plan_minipilot(stage="MP", seeds=MINIPILOT_SEEDS):
    return [_ff_job(stage, ds, arm, s, MINIPILOT_CFG, MINIPILOT_EMAX)
            for ds in DATASETS for arm in models.MINIPILOT_ARMS for s in seeds]


def plan_a(emax, n_points=16):
    pts = sobol_points(n_points)
    jobs = []
    for ds in DATASETS:
        for i, cfg in enumerate(pts):
            jobs += [_ff_job("A", ds, arm, STAGE_A_SEED, cfg, emax, point=i) for arm in models.FF_ARMS]
            jobs += [_bp_job("A", ds, arm, STAGE_A_SEED, cfg, emax, point=i) for arm in models.BP_ARMS]
    return jobs


def load_runs(results, stage):
    runs = []
    for path in sorted(glob.glob(os.path.join(results, stage, "runs", "*.json"))):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def top3(results):
    """Top-3 Stage A points per (dataset, arm) by full validation accuracy (§5.4)."""
    by = defaultdict(list)
    for r in load_runs(results, "A"):
        j = r["job"]
        by[(j["dataset"], j["arm"])].append((r["val_acc"], j["point"], j["cfg"]))
    out = {}
    for key, rows in by.items():
        rows.sort(key=lambda t: (-t[0], t[1]))
        out[key] = [(p, cfg) for _, p, cfg in rows[:3]]
    return out


def plan_b(results, emax):
    jobs = []
    for (ds, arm), cands in sorted(top3(results).items()):
        for p, cfg in cands:
            for s in STAGE_B_SEEDS:
                maker = _bp_job if arm in models.BP_ARMS else _ff_job
                jobs.append(maker("B", ds, arm, s, cfg, emax, point=p))
    return jobs


def _choose(rows):
    """rows: list of (point_id, cfg, mean_val). Highest mean; tie -> lower c (§5.4)."""
    return sorted(rows, key=lambda t: (-t[2], t[1].get("c", 0.0)))[0]


def b_means(results, stages=("B", "B2")):
    acc = defaultdict(list)
    cfgs = {}
    for st in stages:
        for r in load_runs(results, st):
            j = r["job"]
            key = (j["dataset"], j["arm"], str(j["point"]))
            acc[key].append(r["val_acc"])
            cfgs[key] = j["cfg"]
    table = defaultdict(list)
    for (ds, arm, p), vals in acc.items():
        if len(vals) == len(STAGE_B_SEEDS):
            table[(ds, arm)].append((p, cfgs[(ds, arm, p)], float(np.mean(vals))))
    return table


def _edge_axes(choices, axes):
    """Count arms whose chosen config lies in the outer 10% (log) of each axis."""
    counts = {ax: 0 for ax in axes}
    for cfg in choices:
        for ax in axes:
            lo, hi = AXES[ax]
            v = (math.log10(cfg[ax]) - lo) / (hi - lo)
            if v <= 0.1 or v >= 0.9:
                counts[ax] += 1
    edge = [ax for ax in axes if counts[ax] > 0]
    edge.sort(key=lambda ax: (-counts[ax], AXIS_ORDER.index(ax)))
    return edge[:2]


def plan_b2(results, emax):
    """Boundary rule (§5.4): per dataset, edge axes from all arms; every arm gets ±25% points."""
    table = b_means(results, ("B",))
    jobs = []
    for ds in DATASETS:
        arms = [a for (d, a) in table if d == ds]
        chosen = {a: _choose(table[(ds, a)]) for a in arms}
        for kind_arms, axes in ((models.FF_ARMS, ["lr", "wd", "c"]), (list(models.BP_ARMS), ["lr", "wd"])):
            group = [a for a in arms if a in kind_arms]
            edge = _edge_axes([chosen[a][1] for a in group], axes)
            for a in group:
                p0, cfg0, _ = chosen[a]
                seen = {json.dumps(c[1], sort_keys=True) for c in table[(ds, a)]}
                for ax in edge:
                    lo, hi = AXES[ax]
                    for sign in (-1, 1):
                        cfg = dict(cfg0)
                        val = math.log10(cfg0[ax]) + sign * 0.25 * (hi - lo)
                        if ax == "c":
                            val = min(val, 0.0)
                        cfg[ax] = float(10 ** val)
                        key = json.dumps(cfg, sort_keys=True)
                        if key in seen:
                            continue
                        seen.add(key)
                        pid = f"{p0}{ax}{'+' if sign > 0 else '-'}"
                        for s in STAGE_B_SEEDS:
                            maker = _bp_job if a in models.BP_ARMS else _ff_job
                            jobs.append(maker("B2", ds, a, s, cfg, emax, point=pid))
    return jobs


def select_b(results):
    table = b_means(results, ("B", "B2"))
    return {f"{ds}|{arm}": {"point": p, "cfg": cfg, "mean_val": m}
            for (ds, arm), rows in sorted(table.items()) for p, cfg, m in [_choose(rows)]}


def emax_rule(results, stage, emax):
    """Active-limit rule (§5.2): >25% runs of any arm with best epoch >= emax-2 in any layer."""
    by = defaultdict(list)
    for r in load_runs(results, stage):
        active = any(e is not None and e >= emax - 2 for e in r["best_epoch"].values())
        by[(r["job"]["dataset"], r["job"]["arm"])].append(active)
    frac = {f"{k[0]}|{k[1]}": float(np.mean(v)) for k, v in by.items()}
    triggered = any(v > 0.25 for v in frac.values())
    return {"stage": stage, "emax_before": emax, "triggered": triggered,
            "emax_after": int(math.ceil(emax * 1.5)) if triggered else emax, "active_fraction": frac}


def plan_c(choice, n_seeds, emax):
    sel = choice["configs"]
    jobs = []
    for ds in DATASETS:
        for s in range(STAGE_C_SEED0 + 1, STAGE_C_SEED0 + n_seeds + 1):
            jobs += [_ff_job("C", ds, arm, s, sel[f"{ds}|{arm}"]["cfg"], emax) for arm in models.FF_ARMS]
        for s in E_SEEDS:
            jobs += [_bp_job("E2", ds, arm, s, sel[f"{ds}|{arm}"]["cfg"], emax) for arm in models.BP_ARMS]
            for w in models.E9_WIDTHS:
                jobs += [_ff_job("E9", ds, arm, s, sel[f"{ds}|{arm}"]["cfg"], emax, w=w) for arm in models.E9_ARMS]
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--results")
    ap.add_argument("--emax", type=int)
    ap.add_argument("--stage")
    ap.add_argument("--choice")
    ap.add_argument("--n", type=int)
    a = ap.parse_args(argv)
    if a.what == "minipilot":
        obj = plan_minipilot()
    elif a.what == "minipilot2":
        obj = plan_minipilot("MP2", MINIPILOT2_SEEDS)
    elif a.what == "A":
        obj = plan_a(a.emax)
    elif a.what == "B":
        obj = plan_b(a.results, a.emax)
    elif a.what == "B2":
        obj = plan_b2(a.results, a.emax)
    elif a.what == "select-B":
        obj = {"configs": select_b(a.results)}
    elif a.what == "emax-rule":
        print(json.dumps(emax_rule(a.results, a.stage, a.emax), indent=1))
        return
    elif a.what == "C":
        with open(a.choice) as f:
            choice = json.load(f)
        obj = plan_c(choice, a.n, choice["emax"])
    else:
        raise SystemExit(f"unknown: {a.what}")
    with open(a.out, "w") as f:
        json.dump(obj, f, indent=1)
    print(f"{a.what}: {len(obj) if isinstance(obj, list) else len(obj['configs'])} entries -> {a.out}")


if __name__ == "__main__":
    main()
