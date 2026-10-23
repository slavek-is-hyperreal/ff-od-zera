"""Mini-pilot derivation (PROTOKOL_v5 §9.3, §7.5, §5.2, §11.2).

Usage:  python -m ffexp.minipilot --results WYNIKI --price USD_PER_HOUR --workers K --out PARAMETRY.json

Committed BEFORE the mini-pilot is run; executed once on the committed MINIPILOT.json data.
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from . import analysis, models, plans


def _val_preds(results, runs):
    """{run id: (val_idx, correct)} — labels are read from the training set (validation is part of it)."""
    from . import data

    labels = {ds: data.load(ds, "train")[1].numpy() for ds in ("mnist", "fmnist")}
    out = {}
    for r in runs:
        p = torch.load(os.path.join(results, r["job"]["stage"], "preds", r["id"] + ".pt"), weights_only=False)
        idx = p["val_idx"].numpy()
        out[r["id"]] = (idx, p["val_pred"].numpy().astype(np.int64) == labels[r["job"]["dataset"]][idx])
    return out


def forecast(runs, n, emax, price, workers, emax_worst=None):
    """Cost forecast (USD) per stage from measured seconds per epoch (§11.2).

    Seconds per layer-epoch are measured per topology; BP and E9 are conservatively charged
    at the P rate. Wall clock = job-seconds / workers (throughput measured with that k)."""
    sec = {"P": [], "K": []}
    epochs_mean = []
    for r in runs:
        topo = "P" if r["job"]["arm"].startswith("P-") else "K"
        ep = sum(r["epochs_run"].values())
        sec[topo].append(r["wall_time_sec"] / max(ep, 1))
        epochs_mean.append(np.mean(list(r["epochs_run"].values())))
    s_ep = {k: float(np.mean(v)) for k, v in sec.items()}
    ep_layer = float(np.mean(epochs_mean))

    def stage_cost(jobs, ep):
        tot = 0.0
        for j in jobs:
            topo = "K" if j["arm"].startswith("K-") and j["kind"] == "ff" else "P"
            layers = 2 if j["kind"] == "ff" else 1
            tot += s_ep[topo] * ep * layers
        return tot / workers / 3600 * price

    fake_choice = {"configs": {f"{ds}|{a}": {"cfg": {"lr": 1e-3, "wd": 1e-4, "c": 0.25}}
                               for ds in plans.DATASETS for a in models.FF_ARMS + list(models.BP_ARMS)},
                   "emax": emax}
    a_jobs = plans.plan_a(emax)
    n_b = len(models.FF_ARMS + list(models.BP_ARMS)) * len(plans.DATASETS)
    b_runs = n_b * 9 + n_b * 12  # top-3 x 3 seeds + boundary rule (assumed triggered)
    out = {"sec_per_layer_epoch": s_ep, "mean_epochs_per_layer_minipilot": ep_layer}
    for label, ep in (("expected", ep_layer), ("worst", emax_worst or emax)):
        c_jobs = plans.plan_c(fake_choice, n, emax)
        cost = {
            "A": stage_cost(a_jobs, ep),
            "B": stage_cost(a_jobs, ep) / len(a_jobs) * b_runs,
            "C+E2+E9": stage_cost(c_jobs, ep),
        }
        cost["total"] = sum(cost.values())
        out[label] = cost
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--price", type=float, required=True, help="instance price, USD/hour")
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=20000)
    ap.add_argument("--stage", default="MP2", help="MP2 is the only valid source of derivations (decision D4)")
    a = ap.parse_args(argv)
    runs = []
    for path in sorted(glob.glob(os.path.join(a.results, a.stage, "runs", "*.json"))):
        with open(path) as f:
            runs.append(json.load(f))
    expected = len(models.MINIPILOT_ARMS) * len(plans.MINIPILOT_SEEDS) * len(plans.DATASETS)
    if len(runs) != expected:
        raise SystemExit(f"mini-pilot incomplete: {len(runs)}/{expected} runs")
    der = analysis.derive_minipilot(runs, _val_preds(a.results, runs), reps=a.reps)
    worst_emax = int(np.ceil(np.ceil(der["emax"] * 1.5) * 1.5))
    der["budget_usd"] = {str(n): forecast(runs, n, der["emax"], a.price, a.workers, worst_emax)
                         for n in (30, 40, 50, 60)}
    der["S"] = {ds: {"S_H1H2": v["S_H1H2"], "S_H3": v["S_H3"]} for ds, v in der["datasets"].items()}
    der["instance"] = {"price_usd_per_hour": a.price, "workers": a.workers}
    with open(a.out, "w") as f:
        json.dump(der, f, indent=1)
    print(json.dumps({k: der[k] for k in ("N", "emax", "rho", "S")}, indent=1))
    print("koszt (USD) przy N:", {n: round(v["expected"]["total"], 2) for n, v in der["budget_usd"].items()})


if __name__ == "__main__":
    main()
