#!/usr/bin/env python
# coding: utf-8

"""Animate the time evolution of |phi(z)| from mominzt.nc."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-temp_test_dlra")
import matplotlib
if os.environ.get("MPLBACKEND"):
    matplotlib.use(os.environ["MPLBACKEND"])
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate |phi(z)| from mominzt.nc")
    parser.add_argument("--input", default="lingk_output/mominzt.nc")
    parser.add_argument("--interval", type=int, default=100, help="Animation interval in ms")
    parser.add_argument("--every", type=int, default=1, help="Use every N-th saved time slice")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--save", default=None, help="Optional output animation path (.gif/.mp4)")
    parser.add_argument("--frames-dir", default=None, help="Optional directory for PNG frame dumps")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def load_mominzt(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with xr.open_dataset(path) as f:
        time = f["time"].values
        z = f["z"].values
        phi_real = f["phi_real"].values
        phi_imag = f["phi_imag"].values
    phi_abs = np.sqrt(phi_real**2 + phi_imag**2)
    return time, z, phi_abs


def save_frames(
    selected_times: np.ndarray,
    z: np.ndarray,
    selected_phi_abs: np.ndarray,
    frames_dir: Path,
) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for iframe, (time_value, frame) in enumerate(zip(selected_times, selected_phi_abs)):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        positive = frame[frame > 0.0]
        ymin = max(float(np.min(positive)) * 0.8, 1.0e-16) if positive.size else 1.0e-16
        ymax = float(np.max(frame)) * 1.2 if np.max(frame) > 0.0 else 1.0
        ax.plot(z, frame, color="tab:blue", lw=2)
        ax.set_xlabel("z")
        ax.set_ylabel("|phi(z)|")
        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)
        ax.text(
            0.02,
            0.96,
            f"t = {float(time_value):.8f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
        fig.tight_layout()
        fig.savefig(frames_dir / f"mominz_{iframe:04d}.png", dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    time, z, phi_abs = load_mominzt(args.input)

    frame_indices = np.arange(0, len(time), args.every, dtype=int)
    if frame_indices.size == 0:
        raise ValueError("No frames selected. Check --every.")
    selected_times = np.asarray(time[frame_indices], dtype=float)
    selected_phi_abs = np.asarray(phi_abs[frame_indices], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    (line,) = ax.plot([], [], color="tab:blue", lw=2)
    time_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    ax.set_xlabel("z")
    ax.set_ylabel("|phi(z)|")
    ax.set_yscale("log")
    ax.set_xlim(float(z[0]), float(z[-1]))
    ax.grid(True, alpha=0.3)

    def frame_ylim(values: np.ndarray) -> tuple[float, float]:
        positive = values[values > 0.0]
        ymin = max(float(np.min(positive)) * 0.8, 1.0e-16) if positive.size else 1.0e-16
        ymax = float(np.max(values)) * 1.2 if np.max(values) > 0.0 else 1.0
        return ymin, ymax

    def init():
        frame = selected_phi_abs[0]
        line.set_data(z, frame)
        ax.set_ylim(*frame_ylim(frame))
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        time_text.set_text(f"t = {selected_times[0]:.8f}")
        return line, time_text

    def update(frame_number: int):
        frame = selected_phi_abs[frame_number]
        line.set_data(z, frame)
        ax.set_ylim(*frame_ylim(frame))
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        time_text.set_text(f"t = {selected_times[frame_number]:.8f}")
        return line, time_text

    if args.frames_dir is not None:
        save_frames(selected_times, z, selected_phi_abs, Path(args.frames_dir))
        print(f"[plot_mominz] wrote frames: {args.frames_dir}")

    anim = None
    if (not args.no_show) or (args.save is not None):
        anim = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=len(frame_indices),
            interval=args.interval,
            blit=False,
            repeat=False,
        )
    fig.tight_layout()

    if args.save is not None:
        save_path = Path(args.save)
        if anim is None:
            raise RuntimeError("Animation object was not created before save.")
        if save_path.suffix.lower() == ".gif":
            anim.save(save_path, writer=PillowWriter(fps=args.fps))
        else:
            anim.save(save_path, fps=args.fps)
        print(f"[plot_mominz] wrote animation: {save_path}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
