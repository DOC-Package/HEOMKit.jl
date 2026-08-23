# HEOMKit

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://htkhsh.github.io/HEOMKit.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://htkhsh.github.io/HEOMKit.jl/dev/)
[![Build Status](https://github.com/DOC-Package/HEOMKit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/htkhsh/HEOMKit.jl/actions/workflows/CI.yml?query=branch%3Amain)

## BLAS backend selection

HEOMKit itself does not require MKL. It uses Julia's BLAS/LAPACK layer through
`LinearAlgebra`, so the active backend is whichever backend Julia's
`libblastrampoline` is forwarding to.

- **OpenBLAS**: Julia default
- **MKL**: `using MKL` before `using HEOMKit`
- **AOCL**: set `LBT_DEFAULT_LIBS` to your AOCL BLAS/LAPACK libraries before starting Julia

Examples:

```bash
# OpenBLAS (default)
julia --project=.

# MKL
julia --project=. -e 'using MKL; using HEOMKit; println(blas_config())'

# AOCL (adjust library filenames to your installation)
export LBT_DEFAULT_LIBS="/path/to/aocl/lib/libblis.so;/path/to/aocl/lib/libflame.so"
julia --project=. -e 'using HEOMKit; println(blas_config())'
```

You can inspect and tune BLAS threading from HEOMKit:

```julia
using HEOMKit

blas_config()
blas_num_threads()
set_blas_threads!(8)
```
