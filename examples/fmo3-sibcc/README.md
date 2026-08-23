# Three-site FMO62 SIBCC-HEOM example

This example uses the three-site FMO Hamiltonian

```text
[ 310.0  -97.9    5.5 ]
[ -97.9  230.0   30.1 ] cm^-1
[   5.5   30.1    0.0 ]
```

and couples an independent AR + 62-mode FMO bath to each localized site. It
compares site-resolved SIBCC reductions with a direct ESPRIT fit and supports
an independent ESPRIT candidate-generation time grid.

Run from the HEOMKit repository root:

```sh
julia --project=. examples/fmo3-sibcc/heom.jl
julia --project=. examples/fmo3-sibcc/plot_results.jl
```

Direct-ESPRIT tolerance convergence can be computed separately:

```sh
julia --project=. examples/fmo3-sibcc/esprit_convergence.jl
julia --project=. examples/fmo3-sibcc/plot_esprit_convergence.jl
```

The scripts warn when ESPRIT produces poles with non-positive real parts.
