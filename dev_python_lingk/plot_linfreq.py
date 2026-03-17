#!/usr/bin/env python
# coding: utf-8

"""Plot the linear frequency history from frq.txt as a single multi-panel figure."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-temp_test_dlra")
import matplotlib
if os.environ.get("MPLBACKEND"):
    matplotlib.use(os.environ["MPLBACKEND"])
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot growth, frequency, and convergence metrics from frq.txt")
    parser.add_argument("--input", default="lingk_output/frq.txt")
    parser.add_argument("--save", default=None, help="Optional output image path")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def load_linfreq(path: str | Path) -> np.ndarray:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] != 6:
        raise ValueError(f"{path} has {data.shape[1]} columns, expected 6")
    return data


def main() -> None:
    args = parse_args()
    data = load_linfreq(args.input)

    time = data[:, 0]
    growth = data[:, 1]
    frequency = data[:, 2]
    diff_growth = data[:, 3]
    diff_frequency = data[:, 4]
    ineq = data[:, 5]

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), sharex=True)

    axes[0].plot(time, growth, color="tab:blue", marker="o", ms=3, lw=1.5)
    axes[0].set_ylabel("Growthrate")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, frequency, color="tab:orange", marker="o", ms=3, lw=1.5)
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time, diff_growth, label="Diff(grow)", color="tab:green", marker="o", ms=3, lw=1.5)
    axes[2].plot(time, diff_frequency, label="Diff(freq)", color="tab:red", marker="s", ms=3, lw=1.5)
    axes[2].plot(time, ineq, label="1 - Ineq.", color="tab:purple", marker="^", ms=3, lw=1.5)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Convergence")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3, which="both")
    axes[2].legend(loc="best")

    fig.tight_layout()

    if args.save is not None:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
        print(f"[plot_linfreq] wrote figure: {save_path}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
