# KaisouEOM

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://htkhsh.github.io/KaisouEOM.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://htkhsh.github.io/KaisouEOM.jl/dev/)
[![Build Status](https://github.com/DOC-Package/KaisouEOM.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/htkhsh/KaisouEOM.jl/actions/workflows/CI.yml?query=branch%3Amain)

## BLAS backend selection

KaisouEOM itself does not require MKL. It uses Julia's BLAS/LAPACK layer through
`LinearAlgebra`, so the active backend is whichever backend Julia's
`libblastrampoline` is forwarding to.

- **OpenBLAS**: Julia default
- **MKL**: `using MKL` before `using KaisouEOM`
- **AOCL**: set `LBT_DEFAULT_LIBS` to your AOCL BLAS/LAPACK libraries before starting Julia

Examples:

```bash
# OpenBLAS (default)
julia --project=.

# MKL
julia --project=. -e 'using MKL; using KaisouEOM; println(blas_config())'

# AOCL (adjust library filenames to your installation)
export LBT_DEFAULT_LIBS="/path/to/aocl/lib/libblis.so;/path/to/aocl/lib/libflame.so"
julia --project=. -e 'using KaisouEOM; println(blas_config())'
```

You can inspect and tune BLAS threading from KaisouEOM:

```julia
using KaisouEOM

blas_config()
blas_num_threads()
set_blas_threads!(8)
```
