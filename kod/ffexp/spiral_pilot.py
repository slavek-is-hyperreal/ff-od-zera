"""Mini-pilot 2.0, spiral part (decision D4): learning dynamics on pilot seeds 49001-49003.

Usage:  python -m ffexp.spiral_pilot run --out WYNIKI/spirala_pilot [--workers 4]
        python -m ffexp.spiral_pilot plots --out WYNIKI/spirala_pilot

Records, during training, the decision landscape on a grid: score(class 1) - score(class 0)
(FF: goodness; layer-1 phase uses G1, layer-2 phase G1+G2; BP: logit difference).
Uses only the spiral train/validation split; the spiral test split is never touched.
"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np
import torch

from . import data, models, train
from .spiral import EMAX, PATIENCE

SEEDS = [49001, 49002, 49003]
ARMS = ["P-R", "K-LGT", "K-GGT", "BP-P-R"]
WIDTHS = [16, 64]
CFG = {"lr": 3e-3, "wd": 1e-4, "c": 0.05}
GRID_N = 120
FRAME_EVERY = 50
SNAP_EPOCHS = {0, 10, 100, 1000}


def _grid():
    g = torch.linspace(-1.2, 1.2, GRID_N)
    gx, gy = torch.meshgrid(g, g, indexing="xy")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], 1)


@torch.no_grad()
def _margin_ff(l1, l2, layer_no, x):
    task = data.SpiralTask
    s = []
    for c in (0, 1):
        g = models.goodness(l1(task.embed(x, c)))
        if layer_no == 2:
            g = g + models.goodness(l2(models.normalize(l1(task.embed(x, c)))))
        s.append(g)
    return (s[1] - s[0]).reshape(GRID_N, GRID_N).numpy()


@torch.no_grad()
def _margin_bp(net, x):
    logits = net(x)
    return (logits[:, 1] - logits[:, 0]).reshape(GRID_N, GRID_N).numpy()


def run_one(args):
    arm, w, seed, out = args
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sp = data.make_spiral()
    (xtr, ytr), (xva, yva) = sp["train"], sp["val"]
    grid = _grid()
    frames = []  # (phase, epoch, es_acc, margin float16)

    def keep(layer_no, epoch):
        return epoch % FRAME_EVERY == 0 or epoch in SNAP_EPOCHS

    if arm.startswith("BP-"):
        def cb(layer_no, epoch, acc, net):
            if keep(layer_no, epoch):
                frames.append((0, epoch, acc, _margin_bp(net, grid).astype(np.float16)))
        net, info = train.train_bp(arm, data.SpiralTask, xtr, ytr, xva, yva, CFG, seed, EMAX, w=w, spiral=True,
                                   full_batch=True, patience=PATIENCE, callback=cb)
        final = _margin_bp(net, grid)
        layers = {"bp": info["bp"]}
    else:
        def cb(layer_no, epoch, acc, l1, l2):
            if keep(layer_no, epoch):
                frames.append((layer_no, epoch, acc, _margin_ff(l1, l2, layer_no, grid).astype(np.float16)))
        l1, l2, info = train.train_ff(arm, data.SpiralTask, xtr, ytr, xva, yva, CFG, seed, EMAX, w=w, spiral=True,
                                      full_batch=True, patience=PATIENCE, callback=cb)
        final = _margin_ff(l1, l2, 2 if info["layer2_ok"] else 1, grid)
        layers = {"layer1": info["layer1"], "layer2": info["layer2"]}
    name = f"{arm}_w{w}_s{seed}"
    np.savez_compressed(os.path.join(out, "klatki", name + ".npz"),
                        phase=np.array([f[0] for f in frames]), epoch=np.array([f[1] for f in frames]),
                        acc=np.array([f[2] for f in frames]), margin=np.stack([f[3] for f in frames]),
                        final=final.astype(np.float16))
    return {"name": name, "arm": arm, "w": w, "seed": seed, "n_params": info["n_params"],
            "best_epoch": {k: v["best_epoch"] for k, v in layers.items()},
            "epochs_run": {k: v["epochs_run"] for k, v in layers.items()},
            "best_val_acc": {k: v["best_es_acc"] for k, v in layers.items()},
            "curves": {k: [(c["epoch"], c["es_acc"]) for c in v["curve"]] for k, v in layers.items()}}


def run(out, workers):
    os.makedirs(os.path.join(out, "klatki"), exist_ok=True)
    jobs = [(a, w, s, out) for a in ARMS for w in WIDTHS for s in SEEDS]
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(run_one, jobs, chunksize=1)
    hits = [any(e is not None and e >= EMAX - 1000 for e in r["best_epoch"].values()) for r in res]
    rule = {"fraction_best_epoch_ge_emax_minus_1000": float(np.mean(hits)), "threshold": 0.25,
            "emax_now": EMAX, "emax_e10": int(EMAX * 1.5) if np.mean(hits) > 0.25 else EMAX}
    with open(os.path.join(out, "pilot_spirali.json"), "w") as f:
        json.dump({"cfg": CFG, "emax": EMAX, "patience": PATIENCE, "rule_D4": rule, "runs": res}, f, indent=1)
    print(json.dumps(rule))


def plots(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation

    with open(os.path.join(out, "pilot_spirali.json")) as f:
        rec = json.load(f)
    sp = data.make_spiral()
    xtr, ytr = sp["train"]
    ext = (-1.2, 1.2, -1.2, 1.2)
    labels = {"P-R": "FF płaska ReLU (P-R)", "K-LGT": "FF blok L→G→T (K-LGT)",
              "K-GGT": "FF blok G→G→T (K-GGT)", "BP-P-R": "backprop, płaska ReLU"}

    def load(arm, w, seed):
        return np.load(os.path.join(out, "klatki", f"{arm}_w{w}_s{seed}.npz"))

    def draw(ax, margin, title):
        m = margin.astype(np.float32)
        lim = np.percentile(np.abs(m), 98) + 1e-9
        ax.imshow(m, origin="lower", extent=ext, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="bilinear")
        ax.contour(np.linspace(-1.2, 1.2, GRID_N), np.linspace(-1.2, 1.2, GRID_N), m, levels=[0],
                   colors="k", linewidths=0.8)
        ax.scatter(xtr[:, 0], xtr[:, 1], c=ytr, cmap="RdBu_r", s=3, edgecolors="none", alpha=0.6)
        ax.set_title(title, fontsize=7)
        ax.set_xticks([]), ax.set_yticks([])

    # Figure 1: snapshots during training (w=16, first seed)
    w, seed = 16, SEEDS[0]
    cols = [("L1", 0), ("L1", 10), ("L1", 100), ("L1", "koniec"), ("L2", 10), ("L2", 100), ("L2", 1000), ("koniec", None)]
    fig, axes = plt.subplots(len(ARMS), len(cols), figsize=(1.6 * len(cols), 1.75 * len(ARMS)))
    for i, arm in enumerate(ARMS):
        f = load(arm, w, seed)
        ph, ep, acc, mg = f["phase"], f["epoch"], f["acc"], f["margin"]
        for j, (phase, e) in enumerate(cols):
            ax = axes[i, j]
            if phase == "koniec":
                draw(ax, f["final"], "model końcowy")
                continue
            pnum = 0 if arm.startswith("BP") else (1 if phase == "L1" else 2)
            if arm.startswith("BP"):  # BP has a single phase: show monotonically later epochs instead
                e = {("L1", 0): 0, ("L1", 10): 10, ("L1", 100): 100, ("L1", "koniec"): 200,
                     ("L2", 10): 500, ("L2", 100): 1000, ("L2", 1000): "koniec"}[(phase, e)]
            sel = np.where(ph == pnum)[0]
            if e == "koniec":
                k = sel[-1]
            else:
                cand = sel[ep[sel] <= e]
                k = cand[-1] if len(cand) else sel[0]
            draw(ax, mg[k], f"{'BP' if pnum == 0 else 'warstwa ' + str(pnum)}, ep. {ep[k]}\nwal. {acc[k]:.0f}%")
        axes[i, 0].set_ylabel(labels[arm], fontsize=7)
    fig.suptitle("Pilot spirali (D4, ziarno pilotowe): krajobraz decyzji w trakcie uczenia, w=16\n"
                 "kolor: przewaga klasy czerwonej/niebieskiej (FF: różnica dobroci; BP: różnica logitów); "
                 "linia: granica decyzji", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pilot_spirala_klatki.png"), dpi=200)

    # Figure 2: learning curves (all seeds, both widths), cumulative epochs on log scale
    fig, axes = plt.subplots(1, len(WIDTHS), figsize=(9, 3.4), sharey=True)
    colors = {"P-R": "tab:orange", "K-LGT": "tab:blue", "K-GGT": "tab:green", "BP-P-R": "tab:gray"}
    for ax, w in zip(axes, WIDTHS):
        for r in rec["runs"]:
            if r["w"] != w:
                continue
            xs, ys, offset = [], [], 0
            for key in ("layer1", "layer2", "bp"):
                if key in r["curves"]:
                    c = np.array(r["curves"][key])
                    if len(c):
                        xs.extend(c[:, 0] + offset + 1)
                        ys.extend(c[:, 1])
                        if key == "layer1":
                            ax.axvline(offset + len(c), color=colors[r["arm"]], lw=0.4, ls=":")
                        offset += len(c)
            ax.plot(xs, ys, color=colors[r["arm"]],
                    lw=0.8, alpha=0.8, label=labels[r["arm"]] if r["seed"] == SEEDS[0] else None)
        ax.set_xscale("log")
        ax.set_title(f"w={w}", fontsize=8)
        ax.set_xlabel("epoka (łącznie: warstwa 1, potem warstwa 2; kropki = przełączenie)", fontsize=7)
    axes[0].set_ylabel("dokładność walidacyjna (%)")
    axes[0].legend(fontsize=6)
    fig.suptitle("Pilot spirali (D4): krzywe uczenia, 3 ziarna pilotowe", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pilot_spirala_krzywe.png"), dpi=200)

    # Animation: 4 arms side by side, each advancing through its own frames
    fs = {arm: load(arm, 16, SEEDS[0]) for arm in ARMS}
    n_frames = max(len(f["epoch"]) for f in fs.values())
    fig, axes = plt.subplots(1, len(ARMS), figsize=(2.3 * len(ARMS), 2.6))

    def update(t):
        for ax, arm in zip(axes, ARMS):
            ax.clear()
            f = fs[arm]
            k = min(t, len(f["epoch"]) - 1)
            ph = int(f["phase"][k])
            stage = "BP" if ph == 0 else f"warstwa {ph}"
            draw(ax, f["margin"][k], f"{labels[arm]}\n{stage}, ep. {f['epoch'][k]}, wal. {f['acc'][k]:.0f}%")
        return []

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=120)
    anim.save(os.path.join(out, "pilot_spirala.gif"), writer=animation.PillowWriter(fps=8), dpi=90)
    print("rysunki zapisane w", out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["run", "plots"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    if a.what == "run":
        run(a.out, a.workers)
    else:
        plots(a.out)


if __name__ == "__main__":
    main()
