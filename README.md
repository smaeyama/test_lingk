# lingk

`lingk` is a linear gyrokinetic solver for a local flux-tube model.
This repository contains the original Fortran implementation and plotting
scripts for quick inspection with gnuplot.

## Repository layout

- `src/`: Fortran source files for the solver
- `param.namelist`: runtime input parameters
- `Makefile`: build rules for Intel Fortran (`ifx`) and an example GNU Fortran setup
- `plot_*.gn`: gnuplot scripts for inspecting generated output files

## Build

The default `Makefile` uses Intel Fortran:

```bash
make lingk
```

This creates `lingk.exe`.

If you want to use GNU Fortran instead, switch the compiler settings in
[`Makefile`](/home/smaeyama/github/test_lingk/Makefile) by uncommenting the
`gfortran` block and disabling the `ifx` block.

## Run

The solver reads physics parameters from [`param.namelist`](/home/smaeyama/github/test_lingk/param.namelist)
and writes outputs under `./data/`.

```bash
mkdir -p data
./lingk.exe
```

The default sample input includes:

- `ky = 0.2`
- `eps_r = 0.18`
- `q_0 = 1.4`
- `s_hat = 0.8`
- `R0_Ln = 2.2`
- `R0_Lt = 6.9`

## Output files

With the default `flag_runs = 1`, the Fortran solver writes:

- `data/frq.001`: linear growth rate and frequency history
- `data/mominzt.001`: field and density moments as a function of `z` and time
- `data/fkinzv_imXXXX_tYYYYYYYY.dat`: binary snapshots of the distribution
  function for selected `mu` index

The main numerical parameters are defined in
[`src/parameters.f90`](/home/smaeyama/github/test_lingk/src/parameters.f90),
including:

- `nz = 24 * 5`
- `nv = 32`
- `nm = 31`
- `dt_out = 0.1`
- `time_limit = 10.0`

## Quick visualization

If gnuplot is available, the bundled scripts can be used directly:

```bash
gnuplot plot_linfreq.gn
gnuplot plot_mominz.gn
gnuplot plot_fkinzv.gn
```

These scripts expect the default Fortran outputs in `./data/`.
