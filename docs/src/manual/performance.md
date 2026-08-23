# Performance

## Julia threads vs BLAS threads

HEOMKit has two separate layers of parallelism:

1. `parallel=true` enables Julia thread parallelism in the HEOM/HSEOM loops.
2. BLAS/LAPACK uses the backend configured in `LinearAlgebra`.

Set Julia thread count when starting Julia:

```bash
julia -t 8
```

or:

```bash
export JULIA_NUM_THREADS=8
julia
```

Inspect the value in Julia with:

```julia
Threads.nthreads()
```

## Selecting the BLAS backend

HEOMKit does not force MKL. It follows Julia's `libblastrampoline`
configuration, so you can choose the backend outside the package.

### OpenBLAS

OpenBLAS is Julia's default backend.

```bash
julia --project=.
```

### MKL

Load `MKL.jl` before loading HEOMKit:

```julia
using MKL
using HEOMKit
blas_config()
```

### AOCL

AOCL can be selected by forwarding `libblastrampoline` to the AOCL BLAS/LAPACK
libraries before starting Julia.

```bash
export LBT_DEFAULT_LIBS="/path/to/aocl/lib/libblis.so;/path/to/aocl/lib/libflame.so"
julia --project=. -e 'using HEOMKit; println(blas_config())'
```

Exact AOCL library filenames can vary by installation package, so adjust the
paths as needed.

## BLAS thread control

HEOMKit provides small helpers for BLAS thread tuning:

```julia
using HEOMKit

blas_config()
blas_num_threads()
set_blas_threads!(8)
```
