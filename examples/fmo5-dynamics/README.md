# Two-site FMO dynamics with the effective FMO5 bath

This example couples independent copies of the AR + five-Lorentzian effective
FMO bath to the first two localized sites of the standard three-site FMO
Hamiltonian. The 77 K BCF is fitted on 0--5000 fs by direct ESPRIT, and HEOM
construction is aborted if any fitted exponent has a non-positive real part.

Run from the HEOMKit repository root:

```sh
julia --project=. examples/fmo5-dynamics/heom.jl
julia --project=. examples/fmo5-dynamics/plot_results.jl
```
