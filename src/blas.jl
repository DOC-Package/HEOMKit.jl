"""
    blas_config()

Return the current BLAS/LAPACK backend configuration from `LinearAlgebra.BLAS`.

This is useful to confirm whether Julia is currently using OpenBLAS, MKL,
AOCL via `libblastrampoline`, or another supported backend.
"""
blas_config() = BLAS.get_config()

"""
    blas_num_threads() -> Int

Return the current number of BLAS threads.
"""
blas_num_threads() = BLAS.get_num_threads()

"""
    set_blas_threads!(n::Integer) -> Int

Set the number of BLAS threads and return the applied value.
"""
function set_blas_threads!(n::Integer)
    n > 0 || throw(ArgumentError("BLAS thread count must be positive, got $n"))
    BLAS.set_num_threads(Int(n))
    return BLAS.get_num_threads()
end
