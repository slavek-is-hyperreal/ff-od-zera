"""Final blind test evaluation, step 2 of PROTOKOL_v5 §5.6.

Usage:  python -m ffexp.evaluate_test --results WYNIKI --hashes HASHE_CHECKPOINTOW.txt --device cuda

Refuses to run unless every checkpoint of stages C, E2, E9 is listed in the committed
hash file and matches both that file and its run record. Writes per-run test predictions
(main inference, G2-only inference for E6, QMNIST-50k for MNIST models) to
WYNIKI/test/<stage>/<id>.pt.
"""
import argparse
import glob
import json
import os

import torch

from . import data, models, train
from .runner import _sha256

STAGES = ("C", "E2", "E9")


def read_hashes(path):
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                digest, rel = line.split()
                out[rel] = digest
    return out


def _all_ckpts(results):
    """Checkpoints of the confirmatory/exploratory stages, including originals replaced by
    pre-evaluation reruns (kept in <stage>/ckpt_pierwotne, §5.6 pkt 4, §7.8c)."""
    return [ck for st in STAGES for sub in ("ckpt", "ckpt_pierwotne")
            for ck in sorted(glob.glob(os.path.join(results, st, sub, "*.pt")))]


def write_hashes(results, path):
    """Step 1 of §5.6: list sha256 of all checkpoints (to be committed before evaluation)."""
    lines = []
    for ck in _all_ckpts(results):
        lines.append(f"{_sha256(ck)}  {os.path.relpath(ck, results)}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def _rebuild(ckpt, device):
    job, info, state = ckpt["job"], ckpt["info"], ckpt["state"]
    if job["kind"] == "ff":
        l1, l2 = models.build_ff(job["arm"], w=job.get("w", 250))
        l1.load_state_dict(state["layer1"]), l2.load_state_dict(state["layer2"])
        return ("ff", l1.to(device).eval(), l2.to(device).eval()), info
    net = models.BPNet(models.BP_ARMS[job["arm"]], w=job.get("w", 250), in_dim=784)
    net.load_state_dict(state["net"])
    return ("bp", net.to(device).eval()), info


@torch.no_grad()
def _g2_only(l1, l2, x):
    task = data.ImageTask
    scores = torch.zeros(len(x), task.n_classes, device=x.device)
    for c in range(task.n_classes):
        for s in train._chunks(len(x)):
            scores[s, c] = models.goodness(l2(models.normalize(l1(task.embed(x[s], c)))))
    return scores.argmax(1)


def evaluate(results, hashes_path, device="cpu"):
    hashes = read_hashes(hashes_path)
    ckpts = _all_ckpts(results)
    for ck in ckpts:
        rel = os.path.relpath(ck, results)
        if rel not in hashes:
            raise SystemExit(f"checkpoint not in committed hash file: {rel}")
        digest = _sha256(ck)
        stage, sub, name = rel.split(os.sep)
        runs_dir = "runs" if sub == "ckpt" else "runs_pierwotne"
        with open(os.path.join(results, stage, runs_dir, name[:-3] + ".json")) as f:
            run = json.load(f)
        if digest != hashes[rel] or digest != run["ckpt_sha256"]:
            raise SystemExit(f"hash mismatch: {rel}")
    data.enable_test_access()
    tests = {ds: data.load(ds, "test") for ds in ("mnist", "fmnist")}
    tests["qmnist"] = data.load("qmnist", "test50k")
    tests = {k: (x.to(device), y.to(device)) for k, (x, y) in tests.items()}
    for ck in ckpts:
        stage, sub, _ = os.path.relpath(ck, results).split(os.sep)
        out_dir = os.path.join(results, "test", stage if sub == "ckpt" else stage + "_pierwotne")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(ck))
        if os.path.exists(out_path):
            continue
        ckpt = torch.load(ck, map_location="cpu", weights_only=False)
        model, info = _rebuild(ckpt, device)
        ds = ckpt["job"]["dataset"]
        res = {}
        sets = [ds] + (["qmnist"] if ds == "mnist" else [])
        for name in sets:
            x, y = tests[name]
            if model[0] == "ff":
                pred = train.predict_ff(model[1], model[2], data.ImageTask, x, info)
                g2 = _g2_only(model[1], model[2], x) if info["layer1_ok"] and info["layer2_ok"] else pred
                res[name] = {"correct": (pred == y).cpu(), "correct_g2only": (g2 == y).cpu()}
            else:
                pred = train.predict_bp(model[1], x, info)
                res[name] = {"correct": (pred == y).cpu()}
        torch.save(res, out_path + ".tmp")
        os.replace(out_path + ".tmp", out_path)
    return len(ckpts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--hashes", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--write-hashes", action="store_true", help="step 1: write the hash file and exit")
    a = ap.parse_args(argv)
    if a.write_hashes:
        print(f"{write_hashes(a.results, a.hashes)} checkpoints hashed -> {a.hashes}")
        return
    print(f"evaluated {evaluate(a.results, a.hashes, a.device)} checkpoints")


if __name__ == "__main__":
    main()
