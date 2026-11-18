#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download and cache every registered dataset, printing size and timing."""

from __future__ import annotations

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import datasets as ds  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="all")
    args = p.parse_args()
    keys = (ds.TIER_A + ds.TIER_B) if args.datasets == "all" else args.datasets.split(",")
    if args.datasets == "tierA":
        keys = ds.TIER_A
    elif args.datasets == "tierB":
        keys = ds.TIER_B
    fails = []
    for key in keys:
        t0 = time.time()
        try:
            X, y, meta = ds.load(key)
            print("[ok] %-18s N=%-7d d=%-4d K=%-3d classes=%-3d  %.1fs  %s"
                  % (key, meta["N"], meta["d"], meta["K"], meta["K_true"],
                     time.time() - t0, meta["source"]), flush=True)
        except Exception as exc:
            fails.append((key, repr(exc)))
            print("[FAIL] %-16s %r" % (key, exc), flush=True)
    if fails:
        print("\nFAILURES:", flush=True)
        for k, e in fails:
            print("  ", k, e, flush=True)
        return 1
    print("\nall datasets cached", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
