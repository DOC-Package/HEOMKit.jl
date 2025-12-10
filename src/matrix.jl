"""
    Matrix module for HEOM

Liouville 空間における演算子を定義する。
"""

using SparseArrays

# =====================================
# パウリ行列
# =====================================

const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -im; im 0]
const σz = ComplexF64[1 0; 0 -1]
const σI = ComplexF64[1 0; 0 1]
const σp = ComplexF64[0 1; 0 0]   # σ⁺ (raising operator)
const σm = ComplexF64[0 0; 1 0]   # σ⁻ (lowering operator)


# =====================================
# Liouville 空間の演算子（疎行列版）
# =====================================

# 疎行列の型エイリアス
const SparseMat = SparseMatrixCSC{ComplexF64, Int}

"""
    kron_id_left(A::AbstractMatrix)

A ⊗ I を疎行列として計算する（左からの作用）。
"""
function kron_id_left(A::AbstractMatrix)
    n = size(A, 1)
    return kron(sparse(ComplexF64.(A)), sparse(one(ComplexF64) * I, n, n))
end

"""
    kron_id_right(A::AbstractMatrix)

I ⊗ A を疎行列として計算する（右からの作用）。
"""
function kron_id_right(A::AbstractMatrix)
    n = size(A, 1)
    return kron(sparse(one(ComplexF64) * I, n, n), sparse(ComplexF64.(A)))
end

"""
    matx(A::AbstractMatrix)

交換子の Liouville 表現: [A, ρ] → (A ⊗ I - I ⊗ Aᵀ) vec(ρ)
疎行列を返す。
"""
matx(A::AbstractMatrix)::SparseMat = kron_id_left(A) - kron_id_right(transpose(A))

"""
    mato(A::AbstractMatrix)

反交換子の Liouville 表現: {A, ρ} → (A ⊗ I + I ⊗ Aᵀ) vec(ρ)
疎行列を返す。
"""
mato(A::AbstractMatrix)::SparseMat = kron_id_left(A) + kron_id_right(transpose(A))

"""
    matl(A::AbstractMatrix)

左からの作用: Aρ → (A ⊗ I) vec(ρ)
疎行列を返す。
"""
matl(A::AbstractMatrix)::SparseMat = kron_id_left(A)

"""
    matr(A::AbstractMatrix)

右からの作用: ρA† → (I ⊗ conj(A)) vec(ρ)
疎行列を返す。
"""
matr(A::AbstractMatrix)::SparseMat = kron_id_right(conj(A))


# =====================================
# HEOM 行列の構造体
# =====================================

"""
    HEOMMatrices

HEOM 計算に必要な Liouville 空間の行列を保持する構造体。
全ての行列は疎行列（SparseMatrixCSC）として保持される。

# Fields
- `Ls::SparseMat`: システムの Liouvillian -i[H, ·]
- `Vx::Vector{SparseMat}`: 各熱浴の交換子演算子 [V, ·]
- `Vo::Vector{SparseMat}`: 各熱浴の反交換子演算子 {V, ·}
- `Vl::Vector{SparseMat}`: 各熱浴の左作用演算子 V·
- `Vr::Vector{SparseMat}`: 各熱浴の右作用演算子 ·V†
- `NL::Int`: システムのヒルベルト空間次元
- `NL2::Int`: Liouville 空間次元 (NL²)
"""
struct HEOMMatrices
    Ls::SparseMat
    Vx::Vector{SparseMat}
    Vo::Vector{SparseMat}
    Vl::Vector{SparseMat}
    Vr::Vector{SparseMat}
    NL::Int
    NL2::Int
end

"""
    HEOMMatrices(H::AbstractMatrix, noise::Noise)

ハミルトニアン H と Noise から HEOM 行列を構築する。

# Arguments
- `H`: システムハミルトニアン
- `noise`: Noise オブジェクト（熱浴情報を含む）

# Example
```julia
H = [0 1; 1 0]  # 2準位系
noise = Noise([bath1, bath2])
matrices = HEOMMatrices(H, noise)
```
"""
function HEOMMatrices(H::AbstractMatrix, noise::Noise)
    NL = size(H, 1)
    NL2 = NL^2
    
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
    
    return HEOMMatrices(Ls, Vx, Vo, Vl, Vr, NL, NL2)
end


# =====================================
# 表示用
# =====================================

function Base.show(io::IO, m::HEOMMatrices)
    print(io, "HEOMMatrices(NL=$(m.NL), nbath=$(length(m.Vx)))")
end

function Base.show(io::IO, ::MIME"text/plain", m::HEOMMatrices)
    println(io, "HEOMMatrices:")
    println(io, "  System dimension NL = $(m.NL)")
    println(io, "  Liouville dimension NL² = $(m.NL2)")
    println(io, "  Number of baths = $(length(m.Vx))")
end

