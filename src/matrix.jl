"""
    Matrix module for HEOM

Define operators in Liouville space.
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
# Liouville space operators (sparse matrix version)
# =====================================

# Type alias for sparse matrices
const SparseMat = SparseMatrixCSC{ComplexF64, Int}

"""    kron_id_left(A) → A ⊗ I (sparse)"""
function kron_id_left(A::AbstractMatrix)
    n = size(A, 1)
    return kron(sparse(ComplexF64.(A)), sparse(one(ComplexF64) * I, n, n))
end

"""    kron_id_right(A) → I ⊗ A (sparse)"""
function kron_id_right(A::AbstractMatrix)
    n = size(A, 1)
    return kron(sparse(one(ComplexF64) * I, n, n), sparse(ComplexF64.(A)))
end

"""    matx(A) → [A, ρ] in Liouville space"""
matx(A::AbstractMatrix)::SparseMat = kron_id_left(A) - kron_id_right(transpose(A))

"""    mato(A) → {A, ρ} in Liouville space"""
mato(A::AbstractMatrix)::SparseMat = kron_id_left(A) + kron_id_right(transpose(A))

"""    matl(A) → Aρ in Liouville space"""
matl(A::AbstractMatrix)::SparseMat = kron_id_left(A)

"""    matr(A) → ρA† in Liouville space"""
matr(A::AbstractMatrix)::SparseMat = kron_id_right(conj(A))


# =====================================
# HEOM matrices structure
# =====================================

"""
    HEOMMatrices

Liouville space matrices for HEOM: `Ls`, `Vx`, `Vo`, `Vl`, `Vr`, `ndim`, `ndim2`.
"""
struct HEOMMatrices
    Ls::SparseMat
    Vx::Vector{SparseMat}
    Vo::Vector{SparseMat}
    Vl::Vector{SparseMat}
    Vr::Vector{SparseMat}
    ndim::Int
    ndim2::Int
end

"""    HEOMMatrices(H, noise)

Construct HEOM matrices from Hamiltonian and Noise.
"""
function HEOMMatrices(H::AbstractMatrix, noise::Noise)
    ndim = size(H, 1)
    ndim2 = ndim^2
    
    # システム Liouvillian（疎行列）
    Ls = -1.0im * matx(ComplexF64.(H))
    
    # 各熱浴の Liouville 演算子（疎行列）
    nbath = noise.nbath
    Vx = Vector{SparseMat}(undef, nbath)
    Vo = Vector{SparseMat}(undef, nbath)
    Vl = Vector{SparseMat}(undef, nbath)
    Vr = Vector{SparseMat}(undef, nbath)
    
    for ibath in 1:nbath
        V = noise.V[ibath]
        Vx[ibath] = matx(V)
        Vo[ibath] = mato(V)
        Vl[ibath] = matl(V)
        Vr[ibath] = matr(V)
    end
    
    return HEOMMatrices(Ls, Vx, Vo, Vl, Vr, ndim, ndim2)
end


# =====================================
# 表示用
# =====================================

function Base.show(io::IO, m::HEOMMatrices)
    print(io, "HEOMMatrices(ndim=$(m.ndim), nbath=$(length(m.Vx)))")
end

function Base.show(io::IO, ::MIME"text/plain", m::HEOMMatrices)
    println(io, "HEOMMatrices:")
    println(io, "  System dimension ndim = $(m.ndim)")
    println(io, "  Liouville dimension ndim² = $(m.ndim2)")
    println(io, "  Number of baths = $(length(m.Vx))")
end

