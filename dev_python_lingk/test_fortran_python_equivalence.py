from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def ensure_fortran_binary() -> None:
    exe = ROOT / "lingk.exe"
    if exe.exists():
        return
    run(["make", "lingk"], cwd=ROOT)


def have_fortran_reference_outputs() -> bool:
    return (
        (ROOT / "data" / "frq.001").exists()
        and (ROOT / "data" / "mominzt.001").exists()
        and any((ROOT / "data").glob("fkinzv_im*.dat"))
    )


def refresh_fortran_outputs_if_possible() -> None:
    (ROOT / "data").mkdir(exist_ok=True)
    try:
        run(["./lingk.exe"], cwd=ROOT)
    except subprocess.CalledProcessError as exc:
        if have_fortran_reference_outputs():
            return
        stderr = getattr(exc, "stderr", None)
        reason = stderr.decode() if isinstance(stderr, bytes) else str(stderr or exc)
        pytest.skip(f"Fortran executable could not be run and no reference outputs are available: {reason}")


def test_fortran_and_python_outputs_match(tmp_path: Path) -> None:
    ensure_fortran_binary()
    refresh_fortran_outputs_if_possible()

    output_dir = tmp_path / "python_output"
    run(
        [
            sys.executable,
            str(ROOT / "dev_python_lingk" / "lingk.py"),
            "--param-namelist",
            str(ROOT / "param.namelist"),
            "--output-dir",
            str(output_dir),
            "--disable-progress",
            "--time-limit",
            "0.5",
        ],
        cwd=ROOT,
    )

    run(
        [
            sys.executable,
            str(ROOT / "dev_python_lingk" / "check_fortran_vs_python.py"),
            "--python-frq",
            str(output_dir / "frq.txt"),
            "--python-mominzt",
            str(output_dir / "mominzt.nc"),
            "--python-fkinzv",
            str(output_dir / "fkinzv.nc"),
            "--fortran-frq",
            str(ROOT / "data" / "frq.001"),
            "--fortran-mominzt",
            str(ROOT / "data" / "mominzt.001"),
            "--fortran-fkinzv-dir",
            str(ROOT / "data"),
            "--assert-match",
            "--max-abs",
            "1e-5",
            "--max-rel-rms",
            "1e-6",
        ],
        cwd=ROOT,
    )
