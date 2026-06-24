"""
    Matrix module for HEOM

Define operators in Liouville space.
Supports both sparse and dense matrix representations.
"""

using SparseArrays

# =====================================
# Pauli matrices
# =====================================

const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -im; im 0]
const σz = ComplexF64[1 0; 0 -1]
const σI = ComplexF64[1 0; 0 1]
const σp = ComplexF64[0 1; 0 0]   # σ⁺ (raising operator)
const σm = ComplexF64[0 0; 1 0]   # σ⁻ (lowering operator)


# =====================================
# Type aliases
# =====================================

const SparseMat = SparseMatrixCSC{ComplexF64, Int}
const DenseMat = Matrix{ComplexF64}


# =====================================
# Liouville space operators (sparse matrix version)
# =====================================

"""    kron_id_left_sparse(A) → A ⊗ I (sparse)"""
function kron_id_left_sparse(A::AbstractMatrix)
    n = size(A, 1)
    return kron(sparse(ComplexF64.(A)), sparse(one(ComplexF64) * I, n, n))
end

"""    kron_id_right_sparse(A) → I ⊗ A (sparse)"""
function kron_id_right_sparse(A::AbstractMatrix)
    n = size(A, 1)
    return kron(sparse(one(ComplexF64) * I, n, n), sparse(ComplexF64.(A)))
end

"""    matx_sparse(A) → [A, ρ] in Liouville space (sparse)"""
matx_sparse(A::AbstractMatrix)::SparseMat = kron_id_left_sparse(A) - kron_id_right_sparse(transpose(A))

"""    mato_sparse(A) → {A, ρ} in Liouville space (sparse)"""
mato_sparse(A::AbstractMatrix)::SparseMat = kron_id_left_sparse(A) + kron_id_right_sparse(transpose(A))

"""    matl_sparse(A) → Aρ in Liouville space (sparse)"""
matl_sparse(A::AbstractMatrix)::SparseMat = kron_id_left_sparse(A)

"""    matr_sparse(A) → ρA† in Liouville space (sparse)"""
matr_sparse(A::AbstractMatrix)::SparseMat = kron_id_right_sparse(conj(A))


# =====================================
# Liouville space operators (dense matrix version)
# =====================================

"""    kron_id_left_dense(A) → A ⊗ I (dense)"""
function kron_id_left_dense(A::AbstractMatrix)
    n = size(A, 1)
    return kron(Matrix{ComplexF64}(A), Matrix{ComplexF64}(I, n, n))
end

"""    kron_id_right_dense(A) → I ⊗ A (dense)"""
function kron_id_right_dense(A::AbstractMatrix)
    n = size(A, 1)
    return kron(Matrix{ComplexF64}(I, n, n), Matrix{ComplexF64}(A))
end

"""    matx_dense(A) → [A, ρ] in Liouville space (dense)"""
matx_dense(A::AbstractMatrix)::DenseMat = kron_id_left_dense(A) - kron_id_right_dense(transpose(A))

"""    mato_dense(A) → {A, ρ} in Liouville space (dense)"""
mato_dense(A::AbstractMatrix)::DenseMat = kron_id_left_dense(A) + kron_id_right_dense(transpose(A))

"""    matl_dense(A) → Aρ in Liouville space (dense)"""
matl_dense(A::AbstractMatrix)::DenseMat = kron_id_left_dense(A)

"""    matr_dense(A) → ρA† in Liouville space (dense)"""
matr_dense(A::AbstractMatrix)::DenseMat = kron_id_right_dense(conj(A))


# =====================================
# Generic interface (default: sparse)
# =====================================

# Legacy compatibility aliases (use sparse by default)
kron_id_left(A::AbstractMatrix) = kron_id_left_sparse(A)
kron_id_right(A::AbstractMatrix) = kron_id_right_sparse(A)

"""    matx(A) → [A, ρ] in Liouville space"""
matx(A::AbstractMatrix)::SparseMat = matx_sparse(A)

"""    mato(A) → {A, ρ} in Liouville space"""
mato(A::AbstractMatrix)::SparseMat = mato_sparse(A)

"""    matl(A) → Aρ in Liouville space"""
matl(A::AbstractMatrix)::SparseMat = matl_sparse(A)

"""    matr(A) → ρA† in Liouville space"""
matr(A::AbstractMatrix)::SparseMat = matr_sparse(A)


# =====================================
# HEOM matrices structure (abstract type)
# =====================================

"""
    AbstractHEOMMatrices

Abstract type for HEOM Liouville space matrices.
"""
abstract type AbstractHEOMMatrices end

"""
    HEOMMatrices{M<:AbstractMatrix{ComplexF64}}

Liouville space matrices for HEOM.
All V operators are stored per-mode (length = nterms).

# Fields
- `Ls`: System Liouvillian -i[H,·]
- `Vx`: [V,·] per mode
- `Vl`: V· per mode  
- `Vr`: ·V† per mode
- `ndim`: System dimension
- `ndim2`: Liouville dimension (ndim²)
"""
struct HEOMMatrices{M<:AbstractMatrix{ComplexF64}} <: AbstractHEOMMatrices
    Ls::M
    Vx::Vector{M}
    Vl::Vector{M}
    Vr::Vector{M}
    ndim::Int
    ndim2::Int
end

# Type aliases for convenience
const SparseHEOMMatrices = HEOMMatrices{SparseMat}
const DenseHEOMMatrices = HEOMMatrices{DenseMat}

"""    HEOMMatrices(H, noise; sparse=true)

Construct HEOM matrices from Hamiltonian and Noise.
V operators are built per-mode (length = nterms).

# Arguments
- `H`: System Hamiltonian
- `noise`: Noise structure  
- `sparse`: If true (default), use sparse matrices. If false, use dense matrices.
"""
function HEOMMatrices(H::AbstractMatrix, noise::NoiseExp; sparse::Bool=true)
    ndim = size(H, 1)
    ndim2 = ndim^2
    nterms = noise.nterms
    
    if sparse
        # Sparse matrix version
        Ls = -1.0im * matx_sparse(ComplexF64.(H))
        Vx = Vector{SparseMat}(undef, nterms)
        Vl = Vector{SparseMat}(undef, nterms)
        Vr = Vector{SparseMat}(undef, nterms)
        
        for j in 1:nterms
            V = noise.V[j]
            Vx[j] = matx_sparse(V)
            Vl[j] = matl_sparse(V)
            Vr[j] = matr_sparse(V)
        end
        
        return HEOMMatrices{SparseMat}(Ls, Vx, Vl, Vr, ndim, ndim2)
    else
        # Dense matrix version
        Ls = -1.0im * matx_dense(ComplexF64.(H))
        Vx = Vector{DenseMat}(undef, nterms)
        Vl = Vector{DenseMat}(undef, nterms)
        Vr = Vector{DenseMat}(undef, nterms)
        
        for j in 1:nterms
            V = noise.V[j]
            Vx[j] = matx_dense(V)
            Vl[j] = matl_dense(V)
            Vr[j] = matr_dense(V)
        end
        
        return HEOMMatrices{DenseMat}(Ls, Vx, Vl, Vr, ndim, ndim2)
    end
end


# =====================================
# Display functions
# =====================================

function Base.show(io::IO, m::HEOMMatrices{SparseMat})
    print(io, "HEOMMatrices{Sparse}(ndim=$(m.ndim), nterms=$(length(m.Vx)))")
end

function Base.show(io::IO, m::HEOMMatrices{DenseMat})
    print(io, "HEOMMatrices{Dense}(ndim=$(m.ndim), nterms=$(length(m.Vx)))")
end

function Base.show(io::IO, ::MIME"text/plain", m::HEOMMatrices{SparseMat})
    println(io, "HEOMMatrices (Sparse):")
    println(io, "  System dimension ndim = $(m.ndim)")
    println(io, "  Liouville dimension ndim² = $(m.ndim2)")
    println(io, "  Number of terms = $(length(m.Vx))")
end

function Base.show(io::IO, ::MIME"text/plain", m::HEOMMatrices{DenseMat})
    println(io, "HEOMMatrices (Dense):")
    println(io, "  System dimension ndim = $(m.ndim)")
    println(io, "  Liouville dimension ndim² = $(m.ndim2)")
    println(io, "  Number of terms = $(length(m.Vx))")
end

