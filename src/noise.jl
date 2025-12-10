"""
    Noise module for HEOM
"""

"""
    Bath

単一熱浴を表す構造体。熱浴の相関関数をノイズパラメータで展開した係数を保持する。

# Fields
- `expon::Vector{ComplexF64}`: 指数緩和率 γₖ
- `coeff::Vector{ComplexF64}`: 展開係数 cₖ
- `nterms::Int`: ノイズモード数（expon, coeff の長さ）
- `V::Matrix{ComplexF64}`: 系と熱浴の相互作用演算子
"""
struct Bath
    expon::Vector{ComplexF64}
    coeff::Vector{ComplexF64}
    nterms::Int
    V::Matrix{ComplexF64}
end

# Note: Bath(sd::SpectralDensity, ...) constructor is available when QFiND is loaded
# See the package extension or use the direct constructor with gamk/ck vectors.

"""
    Bath(expon::Vector, coeff::Vector, V::AbstractMatrix; add_conjugate=true)

ノイズパラメータを直接指定して Bath を構築する。
複素共役ペアは自動的に追加される。
"""
function Bath(expon::Vector, coeff::Vector, V::AbstractMatrix;
              add_conjugate::Bool=true)
    if add_conjugate
        expon_full = vcat(ComplexF64.(expon), conj.(ComplexF64.(expon)))
        coeff_full = vcat(ComplexF64.(coeff), conj.(ComplexF64.(coeff)))
    else
        expon_full = ComplexF64.(expon)
        coeff_full = ComplexF64.(coeff)
    end
    nterms = length(expon_full)
    return Bath(expon_full, coeff_full, nterms, ComplexF64.(V))
end


# =====================================
# Noise 構造体（複数熱浴を統合）
# =====================================

"""
    Noise

複数の熱浴を統合したノイズパラメータを保持する構造体。

# Fields
- `expon::Vector{ComplexF64}`: 全熱浴の統合 γₖ 配列
- `coeff::Vector{ComplexF64}`: 全熱浴の統合 cₖ 配列
- `abs_coeff::Vector{Float64}`: |cₖ| の配列
- `nterms::Int`: 全ノイズモード数
- `nbath::Int`: 熱浴数
- `nterms_bath::Vector{Int}`: 各熱浴のノイズモード数
- `jstart_bath::Vector{Int}`: 各熱浴の開始インデックス（1-based）
- `V::Vector{Matrix{ComplexF64}}`: 各熱浴の相互作用演算子
- `baths::Vector{Bath}`: 元の Bath オブジェクト
"""
struct Noise
    expon::Vector{ComplexF64}
    coeff::Vector{ComplexF64}
    abs_coeff::Vector{Float64}
    nterms::Int
    nbath::Int
    nterms_bath::Vector{Int}
    jstart_bath::Vector{Int}
    V::Vector{Matrix{ComplexF64}}
    baths::Vector{Bath}
end

"""
    Noise(baths::Vector{Bath})

複数の Bath から Noise を構築する。

# Example
```julia
bath1 = Bath(sd1, 300.0, V1; degree=10)
bath2 = Bath(sd2, 300.0, V2; degree=10)
noise = Noise([bath1, bath2])
```
"""
function Noise(baths::Vector{Bath})
    nbath = length(baths)
    
    # 各熱浴のパラメータサイズと開始位置
    nterms_bath = [b.nterms for b in baths]
    jstart_bath = ones(Int, nbath)
    for i in 2:nbath
        jstart_bath[i] = jstart_bath[i-1] + nterms_bath[i-1]
    end
    
    # 統合配列
    nterms = sum(nterms_bath)
    expon = Vector{ComplexF64}(undef, nterms)
    coeff = Vector{ComplexF64}(undef, nterms)
    
    for (ibath, bath) in enumerate(baths)
        idx_start = jstart_bath[ibath]
        idx_end = idx_start + nterms_bath[ibath] - 1
        expon[idx_start:idx_end] = bath.expon
        coeff[idx_start:idx_end] = bath.coeff
    end
    
    abs_coeff = abs.(coeff)
    V = [b.V for b in baths]
    
    return Noise(expon, coeff, abs_coeff, nterms, nbath, nterms_bath, jstart_bath, V, baths)
end

"""
    Noise(bath::Bath)

単一の Bath から Noise を構築する。
"""
function Noise(bath::Bath)
    return Noise([bath])
end


# =====================================
# 便利なコンストラクタ
# =====================================

"""
    drude_bath(λ::Real, γ::Real, temperature::Real, V::AbstractMatrix; 
               degree::Int=10, integrator::Symbol=:pade)

Drude-Lorentz スペクトル密度を持つ熱浴を作成する。

J(ω) = 2λγω / (ω² + γ²)

# Arguments
- `λ`: 再構成エネルギー
- `γ`: カットオフ周波数
- `temperature`: 温度 [K]
- `V`: 相互作用演算子
- `degree`: 展開次数
"""
function drude_bath(λ::Real, γ::Real, temperature::Real, V::AbstractMatrix;
                    degree::Int=10, integrator::Symbol=:pade)
    sd = DrudeLorentz(λ=λ, γ=γ)
    return Bath(sd, temperature, V; degree=degree, integrator=integrator)
end

"""
    brownian_bath(λ::Real, γ::Real, Ω::Real, temperature::Real, V::AbstractMatrix;
                  degree::Int=10, integrator::Symbol=:pade)

Brownian (underdamped) スペクトル密度を持つ熱浴を作成する。

J(ω) = 4λγΩ²ω / ((ω² - Ω²)² + 4γ²ω²)

# Arguments
- `λ`: 結合強度
- `γ`: 減衰率
- `Ω`: 振動子周波数
- `temperature`: 温度 [K]
- `V`: 相互作用演算子
"""
function brownian_bath(λ::Real, γ::Real, Ω::Real, temperature::Real, V::AbstractMatrix;
                       degree::Int=10, integrator::Symbol=:pade)
    sd = Brownian(λ=λ, γ=γ, Ω=Ω)
    return Bath(sd, temperature, V; degree=degree, integrator=integrator)
end


# =====================================
# 派生パラメータの計算（HEOM用）
# =====================================

"""
    compute_heom_params(noise::Noise)

HEOM 計算に必要な派生パラメータを計算する。

# Returns
- `bk`: real(γₖ) + |imag(γₖ)|
- `ak`: 指数（デフォルト 0.5）
- `cb`: |cₖ| / bₖ
"""
function compute_heom_params(noise::Noise)
    nterms = noise.nterms
    
    bk = [real(noise.expon[j]) + abs(imag(noise.expon[j])) for j in 1:nterms]
    ak = 0.5
    
    cb = [abs(noise.coeff[j]) / bk[j] for j in 1:nterms]
    
    # 正規化
    max_cb = maximum(cb)
    if max_cb > 1.0
        bk .*= max_cb
        cb ./= max_cb
    end
    
    return bk, ak, cb
end


# =====================================
# 表示用
# =====================================

function Base.show(io::IO, bath::Bath)
    print(io, "Bath(nterms=$(bath.nterms))")
end

function Base.show(io::IO, ::MIME"text/plain", bath::Bath)
    println(io, "Bath:")
    println(io, "  nterms = $(bath.nterms)")
    println(io, "  V size = $(size(bath.V))")
    println(io, "  γₖ range: $(minimum(abs.(bath.expon))) - $(maximum(abs.(bath.expon)))")
    println(io, "  cₖ range: $(minimum(abs.(bath.coeff))) - $(maximum(abs.(bath.coeff)))")
end

function Base.show(io::IO, noise::Noise)
    print(io, "Noise(nterms=$(noise.nterms), nbath=$(noise.nbath))")
end

function Base.show(io::IO, ::MIME"text/plain", noise::Noise)
    println(io, "Noise:")
    println(io, "  Total nterms = $(noise.nterms)")
    println(io, "  Number of baths = $(noise.nbath)")
    for i in 1:noise.nbath
        println(io, "    Bath $i: nterms = $(noise.nterms_bath[i]), start = $(noise.jstart_bath[i])")
    end
end