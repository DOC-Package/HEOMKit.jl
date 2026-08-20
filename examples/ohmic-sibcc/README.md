# Ohmic exponential-cutoff SIBCC validation

This minimal example tests SIBCC for a two-level system coupled through
`sigma_z` to one Ohmic bath with an exponential cutoff. It compares a
system-weighted SIBCC reduction against a direct ESPRIT fit using identical
HEOM settings.

Run from the KaisouEOM repository root:

```sh
julia --project=. examples/ohmic-sibcc/heom.jl
julia --project=. examples/ohmic-sibcc/plot_results.jl
```

The numerical summary reports BCF error, system-weighted error, hierarchy
size, final populations, and the maximum population difference.
