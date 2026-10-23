"""Local confirmation of synced runs (PROTOKOL_v5 §8): a run counts as finished only when its
local checkpoint matches the sha256 in its run record.

Usage:  python -m ffexp.sync_verify WYNIKI
"""
import glob
import json
import os
import sys

from .runner import _sha256


def main(argv=None):
    results = (argv or sys.argv[1:])[0]
    confirmed, bad = [], []
    for path in sorted(glob.glob(os.path.join(results, "*", "runs", "*.json"))):
        with open(path) as f:
            run = json.load(f)
        stage = os.path.basename(os.path.dirname(os.path.dirname(path)))
        ck = os.path.join(results, stage, "ckpt", run["id"] + ".pt")
        if os.path.exists(ck) and _sha256(ck) == run["ckpt_sha256"]:
            confirmed.append(f"{stage}/{run['id']}")
        else:
            bad.append(f"{stage}/{run['id']}")
    with open(os.path.join(results, "potwierdzone_lokalnie.txt"), "w") as f:
        f.write("\n".join(confirmed) + "\n")
    print(f"potwierdzone lokalnie: {len(confirmed)}  niezgodne/brak: {len(bad)}")
    for b in bad[:20]:
        print("  ", b)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
