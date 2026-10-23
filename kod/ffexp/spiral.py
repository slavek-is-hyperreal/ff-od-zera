"""E10: 2D spiral, exploratory illustration (PROTOKOL_v5 §3.4, §4.4, §5.7, §7.7, §7.9).

Usage:  python -m ffexp.spiral train --out WYNIKI/E10     (tuning + 10 evaluation seeds; no test access)
        python -m ffexp.spiral final --out WYNIKI/E10     (test accuracy, figures; after all training)
"""
import argparse
import json
import os

import numpy as np
import torch

from . import data, models, train

TUNE_SEED = 50000
EVAL_SEEDS = list(range(50001, 50011))
EMAX, PATIENCE = 5000, 1000  # changed before freeze after a feasibility check (PROTOKOL_v5 §5.7)
FIG_BUDGET = 16
ARMS = models.E10_ARMS + ["BP-P-R"]


def sobol8():
    from scipy.stats import qmc

    from .plans import AXES

    u = qmc.Sobol(d=3, scramble=True, seed=0).random(8)
    return [{ax: float(10 ** (lo + (hi - lo) * v)) for (ax, (lo, hi)), v in zip(AXES.items(), row)} for row in u]


def _fit(arm, w, cfg, seed, sp):
    xtr, ytr = sp["train"]
    xva, yva = sp["val"]
    if arm.startswith("BP-"):
        net, info = train.train_bp(arm, data.SpiralTask, xtr, ytr, xva, yva, cfg, seed, EMAX, w=w, spiral=True,
                                   full_batch=True, patience=PATIENCE)
        model = ("bp", net)
        val = (train.predict_bp(net, xva, info) == yva).float().mean().item() * 100
    else:
        l1, l2, info = train.train_ff(arm, data.SpiralTask, xtr, ytr, xva, yva, cfg, seed, EMAX, w=w, spiral=True,
                                      full_batch=True, patience=PATIENCE)
        model = ("ff", l1, l2)
        val = (train.predict_ff(l1, l2, data.SpiralTask, xva, info) == yva).float().mean().item() * 100
    return model, info, val


def _predict(model, info, x):
    if model[0] == "bp":
        return train.predict_bp(model[1], x, info)
    return train.predict_ff(model[1], model[2], data.SpiralTask, x, info)


def _train_key(args):
    """Tuning (8 points, seed 50000) and 10 evaluation seeds for one (arm, budget)."""
    arm, w, out = args
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    sp = data.make_spiral()
    points = sobol8()
    vals = [_fit(arm, w, cfg, TUNE_SEED, sp)[2] for cfg in points]
    best = int(np.argmax(vals))
    evals = {}
    for s in EVAL_SEEDS:
        model, info, val = _fit(arm, w, points[best], s, sp)
        state = {"info": info, "kind": model[0], "state": [m.state_dict() for m in model[1:]]}
        torch.save(state, os.path.join(out, "modele", f"{arm}_w{w}_s{s}.pt"))
        evals[s] = {"val_acc": val, "n_params": info["n_params"]}
    print(f"{arm}|{w} gotowe", flush=True)
    return f"{arm}|{w}", {"val": vals, "chosen": best, "cfg": points[best]}, evals


def run_train(out, workers=1):
    import multiprocessing as mp

    os.makedirs(os.path.join(out, "modele"), exist_ok=True)
    keys = [(arm, w, out) for arm in ARMS for w in models.E10_WIDTHS]
    with mp.get_context("spawn").Pool(workers) as pool:
        results = pool.map(_train_key, keys, chunksize=1)
    record = {"tuning": {k: tune for k, tune, _ in results}, "eval": {k: ev for k, _, ev in results}}
    with open(os.path.join(out, "e10_trening.json"), "w") as f:
        json.dump(record, f, indent=1)


def _load(out, arm, w, s):
    st = torch.load(os.path.join(out, "modele", f"{arm}_w{w}_s{s}.pt"), weights_only=False)
    if st["kind"] == "bp":
        net = models.BPNet(models.BP_ARMS[arm], n_classes=2, w=w, in_dim=2, spiral=True)
        net.load_state_dict(st["state"][0])
        return ("bp", net), st["info"]
    l1, l2 = models.build_ff(arm, w=w, in_dim=4, spiral=True)
    l1.load_state_dict(st["state"][0]), l2.load_state_dict(st["state"][1])
    return ("ff", l1, l2), st["info"]


def median_seed(evals):
    """Lower median of validation accuracy: 5th of 10 in ascending order; ties -> lower seed (§7.7)."""
    ranked = sorted(evals.items(), key=lambda kv: (kv[1]["val_acc"], int(kv[0])))
    return int(ranked[len(ranked) // 2 - 1][0])


def run_final(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(out, "e10_trening.json")) as f:
        record = json.load(f)
    sp = data.make_spiral()
    xte, yte = sp["test"]
    g = torch.linspace(-1.2, 1.2, 300)
    gx, gy = torch.meshgrid(g, g, indexing="xy")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)
    summary = {}
    for key, evals in record["eval"].items():
        arm, w = key.split("|")
        accs = {}
        for s in evals:
            model, info = _load(out, arm, int(w), int(s))
            accs[s] = (_predict(model, info, xte) == yte).float().mean().item() * 100
        n_params = next(iter(evals.values()))["n_params"]
        summary[key] = {"test_acc": accs, "median_test": float(np.median(list(accs.values()))),
                        "mean_test": float(np.mean(list(accs.values()))), "n_params": n_params,
                        "figure_seed": median_seed(evals)}
    min_budget = {}
    for arm in ARMS:
        ok = [int(key.split("|")[1]) for key, v in summary.items() if key.startswith(arm + "|") and v["median_test"] >= 95]
        min_budget[arm] = min(ok) if ok else None
    with open(os.path.join(out, "e10_wyniki.json"), "w") as f:
        json.dump({"summary": summary, "min_budget_median_ge_95": min_budget}, f, indent=1)

    def panel(ax, arm, w):
        key = f"{arm}|{w}"
        s = summary[key]["figure_seed"]
        model, info = _load(out, arm, w, s)
        pred = _predict(model, info, grid).reshape(300, 300).numpy()
        ax.imshow(pred, origin="lower", extent=(-1.2, 1.2, -1.2, 1.2), cmap="coolwarm", alpha=0.35,
                  vmin=0, vmax=1, interpolation="nearest")
        ax.scatter(xte[:, 0], xte[:, 1], c=yte, cmap="coolwarm", s=6, edgecolors="k", linewidths=0.2)
        ax.set_title(f"{arm}, w={w}, {summary[key]['n_params']} par.\nziarno {s}: test "
                     f"{summary[key]['test_acc'][str(s)]:.1f}%", fontsize=7)
        ax.set_xticks([]), ax.set_yticks([])

    fig, axes = plt.subplots(1, len(ARMS), figsize=(2.2 * len(ARMS), 2.6))
    for ax, arm in zip(axes, ARMS):
        panel(ax, arm, FIG_BUDGET)
    fig.suptitle(f"E10 (eksploracyjne): mapy decyzyjne, budżet w={FIG_BUDGET}; "
                 "ziarno o dolnej medianie dokładności walidacyjnej spośród 10", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rysunek1_mapy_w16.png"), dpi=200)

    fig, axes = plt.subplots(len(models.E10_WIDTHS), len(ARMS), figsize=(2.2 * len(ARMS), 2.4 * 4))
    for i, w in enumerate(models.E10_WIDTHS):
        for j, arm in enumerate(ARMS):
            panel(axes[i, j], arm, w)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rysunekS_mapy_wszystkie.png"), dpi=150)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    for arm in ARMS:
        xs, ms, lo, hi = [], [], [], []
        for w in models.E10_WIDTHS:
            v = summary[f"{arm}|{w}"]
            a = np.array(list(v["test_acc"].values()))
            se = a.std(ddof=1) / np.sqrt(len(a))
            xs.append(v["n_params"]), ms.append(a.mean()), lo.append(a.mean() - 2.262 * se), hi.append(a.mean() + 2.262 * se)
        ax.plot(xs, ms, marker="o", label=arm)
        ax.fill_between(xs, lo, hi, alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("liczba parametrów")
    ax.set_ylabel("dokładność testowa (%)")
    ax.set_title("E10 (eksploracyjne): spirala, średnia ± 95% PU, 10 ziaren", fontsize=8)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "rysunek_e10_krzywe.png"), dpi=200)
    print(json.dumps(min_budget))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["train", "final"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    if a.what == "train":
        run_train(a.out, a.workers)
    else:
        run_final(a.out)


if __name__ == "__main__":
    main()
