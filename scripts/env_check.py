# -*- coding: utf-8 -*-
"""打印实验环境信息（写进论文"实验环境"一节，也用于服务器自检）。"""

from __future__ import annotations

import platform
import sys


def main():
    print("python %s (%s)" % (sys.version.split()[0], platform.platform()))
    for mod in ["numpy", "scipy", "sklearn", "pandas", "openpyxl", "matplotlib"]:
        try:
            m = __import__(mod)
            print("%-10s %s" % (mod, getattr(m, "__version__", "?")))
        except Exception as exc:
            print("%-10s 未安装（%s）" % (mod, exc))
    try:
        import numpy as np
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
        print("BLAS       %s %s" % (blas.get("name"), blas.get("version")))
    except Exception:
        pass
    try:
        import os
        print("CPU 核数   %d" % os.cpu_count())
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    print("内存       %.1f GB" % (int(line.split()[1]) / 1024 / 1024))
                    break
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
