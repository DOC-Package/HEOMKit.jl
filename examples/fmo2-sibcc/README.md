# Two-site FMO62 SIBCC-HEOM example

This example takes the upper-left 2-by-2 block of the three-site FMO
Hamiltonian, couples one independent AR + 62-mode bath to each localized site,
and constructs a separate system-weighted SIBCC exponential expansion for
each site before running HEOM dynamics. For comparison, the same reference
BCF is also fitted directly by ESPRIT and propagated with identical HEOM
settings. The output includes BCF-error, hierarchy-size, and population
comparisons between the two methods.

Run from the HEOMKit repository root:

```sh
julia --project=. examples/fmo2-sibcc/heom.jl
```

The FMO62 spectral-density implementation is loaded from
`QFiND/examples/FMO62/spectral_density.jl`. The initial comparison uses a
0--500 fs SIBCC grid with 251 memory points and 251 output-time points
(`dtau = 2 fs`). Results are written next to `heom.jl`.

The ESPRIT candidate grid is configured independently with
`CANDIDATE_T_MIN`, `CANDIDATE_T_MAX`, and `N_CANDIDATE_SAMPLES` in `heom.jl`.
The script warns whenever candidate or direct-ESPRIT poles have non-positive
real parts; such poles are excluded from SIBCC candidate selection.
Plot generation is disabled by default to keep the first run lightweight; set
`GENERATE_PLOTS = true` after the numerical comparison has completed.

Direct-ESPRIT convergence with respect to its fitting tolerance is computed
and plotted separately:

```sh
julia --project=. examples/fmo2-sibcc/esprit_convergence.jl
julia --project=. examples/fmo2-sibcc/plot_esprit_convergence.jl
```
