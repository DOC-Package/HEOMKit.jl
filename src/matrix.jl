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

Liouville space matrices for HEOM: `Ls`, `Vx`, `Vo`, `Vl`, `Vr`, `ndim`, `ndim2`.
Parametric type M can be SparseMat or DenseMat.
"""
struct HEOMMatrices{M<:AbstractMatrix{ComplexF64}} <: AbstractHEOMMatrices
    Ls::M
    Vx::Vector{M}
    Vo::Vector{M}
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

# Arguments
- `H`: System Hamiltonian
- `noise`: Noise structure
- `sparse`: If true (default), use sparse matrices. If false, use dense matrices.
"""
function HEOMMatrices(H::AbstractMatrix, noise::Noise; sparse::Bool=true)
    ndim = size(H, 1)
    ndim2 = ndim^2
    nbath = noise.nbath
    
    if sparse
        # 疎行列版
        Ls = -1.0im * matx_sparse(ComplexF64.(H))
        Vx = Vector{SparseMat}(undef, nbath)
        Vo = Vector{SparseMat}(undef, nbath)
        Vl = Vector{SparseMat}(undef, nbath)
        Vr = Vector{SparseMat}(undef, nbath)
        
        for ibath in 1:nbath
            V = noise.V[ibath]
            Vx[ibath] = matx_sparse(V)
            Vo[ibath] = mato_sparse(V)
            Vl[ibath] = matl_sparse(V)
            Vr[ibath] = matr_sparse(V)
        end
        
        return HEOMMatrices{SparseMat}(Ls, Vx, Vo, Vl, Vr, ndim, ndim2)
    else
        # 密行列版
        Ls = -1.0im * matx_dense(ComplexF64.(H))
        Vx = Vector{DenseMat}(undef, nbath)
        Vo = Vector{DenseMat}(undef, nbath)
        Vl = Vector{DenseMat}(undef, nbath)
        Vr = Vector{DenseMat}(undef, nbath)
        
        for ibath in 1:nbath
            V = noise.V[ibath]
            Vx[ibath] = matx_dense(V)
            Vo[ibath] = mato_dense(V)
            Vl[ibath] = matl_dense(V)
            Vr[ibath] = matr_dense(V)
        end
        
        return HEOMMatrices{DenseMat}(Ls, Vx, Vo, Vl, Vr, ndim, ndim2)
    end
end


# =====================================
# 表示用
# =====================================

function Base.show(io::IO, m::HEOMMatrices{SparseMat})
    print(io, "HEOMMatrices{Sparse}(ndim=$(m.ndim), nbath=$(length(m.Vx)))")
end

function Base.show(io::IO, m::HEOMMatrices{DenseMat})
    print(io, "HEOMMatrices{Dense}(ndim=$(m.ndim), nbath=$(length(m.Vx)))")
end

function Base.show(io::IO, ::MIME"text/plain", m::HEOMMatrices{SparseMat})
    println(io, "HEOMMatrices (Sparse):")
    println(io, "  System dimension ndim = $(m.ndim)")
    println(io, "  Liouville dimension ndim² = $(m.ndim2)")
    println(io, "  Number of baths = $(length(m.Vx))")
end

function Base.show(io::IO, ::MIME"text/plain", m::HEOMMatrices{DenseMat})
    println(io, "HEOMMatrices (Dense):")
    println(io, "  System dimension ndim = $(m.ndim)")
    println(io, "  Liouville dimension ndim² = $(m.ndim2)")
    println(io, "  Number of baths = $(length(m.Vx))")
end

