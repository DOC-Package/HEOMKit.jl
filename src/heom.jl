"""
    HEOM module

HEOM の構造体と Liouville 演算子を定義する。
"""

# =====================================
# HEOM 演算子の構造体
# =====================================

"""
    HEOMOperators

HEOM 時間発展に必要な演算子を保持する構造体。

# Fields
- `ngamma::Vector{ComplexF64}`: 各 ADO の減衰率 Σₖ nₖ γₖ
- `phi::Matrix{ComplexF64}`: 前進接続係数 √((nₖ+1)|cₖ|)
- `theta_l::Matrix{ComplexF64}`: 後退接続係数（左）
- `theta_r::Matrix{ComplexF64}`: 後退接続係数（右）
"""
struct HEOMOperators
    ngamma::Vector{ComplexF64}
    phi::Matrix{ComplexF64}
    theta_l::Matrix{ComplexF64}
    theta_r::Matrix{ComplexF64}
end

"""
    HEOMOperators(noise::Noise, ado_idx::Matrix{Int}, nado::Int)

Noise と階層インデックスから HEOM 演算子を構築する。

# Arguments
- `noise`: Noise オブジェクト
- `ado_idx`: 階層インデックスベクトル (jmax × nado)
- `nado`: ADO の総数
"""
function HEOMOperators(noise::Noise, ado_idx::Matrix{Int}, nado::Int)
    nterms = noise.nterms
    nbath = noise.nbath
    
    # ngamma: Σₖ nₖ γₖ
    ngamma = Vector{ComplexF64}(undef, nado)
    for n in 1:nado
        ngamma[n] = sum(ado_idx[j, n] * noise.expon[j] for j in 1:nterms)
    end
    
    # phi: √((nₖ+1)|cₖ|)
    phi = Matrix{ComplexF64}(undef, nterms, nado)
    for n in 1:nado
        for j in 1:nterms
            phi[j, n] = sqrt((ado_idx[j, n] + 1) * noise.abs_coeff[j])
        end
    end
    
    # theta_l, theta_r: 熱浴ごとに前半/後半で分ける
    theta_l = zeros(ComplexF64, nterms, nado)
    theta_r = zeros(ComplexF64, nterms, nado)
    
    if nbath == 1
        # 単一熱浴: 全体として前半/後半
        nterms2 = nterms ÷ 2
        for n in 1:nado
            for j in 1:nterms2
                if ado_idx[j, n] > 0 && noise.abs_coeff[j] > 0
                    theta_l[j, n] = sqrt(ado_idx[j, n] / noise.abs_coeff[j]) * noise.coeff[j]
                end
            end
            for j in (nterms2 + 1):nterms
                if ado_idx[j, n] > 0 && noise.abs_coeff[j] > 0
                    theta_r[j, n] = -sqrt(ado_idx[j, n] / noise.abs_coeff[j]) * noise.coeff[j]
                end
            end
        end
    else
        # 複数熱浴: 各熱浴内で前半/後半
        for n in 1:nado
            for ibath in 1:nbath
                jstart = noise.jstart_bath[ibath]
                nterms_b = noise.nterms_bath[ibath]
                jmid = jstart + nterms_b ÷ 2 - 1
                jend = jstart + nterms_b - 1
                
                # 前半（元データ）
                for j in jstart:jmid
                    if ado_idx[j, n] > 0 && noise.abs_coeff[j] > 0
                        theta_l[j, n] = sqrt(ado_idx[j, n] / noise.abs_coeff[j]) * noise.coeff[j]
                    end
                end
                # 後半（複素共役データ）
                for j in (jmid + 1):jend
                    if ado_idx[j, n] > 0 && noise.abs_coeff[j] > 0
                        theta_r[j, n] = -sqrt(ado_idx[j, n] / noise.abs_coeff[j]) * noise.coeff[j]
                    end
                end
            end
        end
    end
    
    return HEOMOperators(ngamma, phi, theta_l, theta_r)
end


# =====================================
# HEOM システム全体
# =====================================

"""
    HEOMSystem

HEOM 計算に必要な全情報を保持する構造体。

# Fields
- `noise::Noise`: ノイズパラメータ
- `matrices::HEOMMatrices`: Liouville 空間の行列
- `operators::HEOMOperators`: HEOM 演算子
- `ado_idx::Matrix{Int}`: 階層インデックス
- `idx_plus::Matrix{Int}`: 前進接続インデックス
- `idx_minus::Matrix{Int}`: 後退接続インデックス
- `nado::Int`: ADO 総数
- `NL::Int`: ヒルベルト空間次元
- `NL2::Int`: Liouville 空間次元
"""
struct HEOMSystem
    noise::Noise
    matrices::HEOMMatrices
    operators::HEOMOperators
    ado_idx::Matrix{Int}
    idx_plus::Matrix{Int}
    idx_minus::Matrix{Int}
    nado::Int
    NL::Int
    NL2::Int
end

"""
    HEOMSystem(H::AbstractMatrix, noise::Noise, ndepth::Int; method::Symbol=:depth)

HEOM システムを構築する。

# Arguments
- `H`: システムハミルトニアン
- `noise`: Noise オブジェクト
- `ndepth`: 階層の深さ
- `method`: 階層構築方法 (:depth または :width)

# Example
```julia
H = [0 100; 100 0]  # 2準位系
noise = Noise(bath)
system = HEOMSystem(H, noise, 5)
```
"""
function HEOMSystem(H::AbstractMatrix, noise::Noise, ndepth::Int;
                    hierarchy::Symbol=:depth)
    # 行列を構築
    matrices = HEOMMatrices(H, noise)
    
    # 階層インデックスを構築
    nterms = noise.nterms
    if hierarchy == :depth
        nado, ado_idx, idx_plus, idx_minus = hierarchy_index_depth(nterms, ndepth)
    elseif hierarchy == :width
        bk, ak, _ = compute_heom_params(noise)
        nado, ado_idx, idx_plus, idx_minus = hierarchy_index_width(
            nterms, ndepth, noise.expon, noise.coeff, bk;
            ak=ak, filter=false
        )
    else
        error("Unknown method: $method. Use :depth or :width")
    end
    
    # 演算子を構築
    operators = HEOMOperators(noise, ado_idx, nado)
    
    return HEOMSystem(noise, matrices, operators, ado_idx, idx_plus, idx_minus,
                      nado, matrices.NL, matrices.NL2)
end


# =====================================
# Liouville 演算子
# =====================================

"""
    liouville!(dP::Matrix{ComplexF64}, P::Matrix{ComplexF64}, system::HEOMSystem)

HEOM の Liouville 演算子を適用する（in-place）。

dρₙ/dt = -i[H, ρₙ] - Σₖ nₖγₖ ρₙ 
         - i Σₖ [V, √((nₖ+1)|cₖ|) ρₙ₊ₖ]
         - i Σₖ (Vρₙ₋ₖ θₗ - ρₙ₋ₖ V† θᵣ)
"""
function liouville!(dP::Matrix{ComplexF64}, P::Matrix{ComplexF64}, system::HEOMSystem)
    (; matrices, operators, noise, ado_idx, idx_plus, idx_minus, nado, NL2) = system
    (; Ls, Vx, Vl, Vr) = matrices
    (; ngamma, phi, theta_l, theta_r) = operators
    nterms = noise.nterms
    nbath = noise.nbath
    
    dP .= 0.0 + 0.0im
    
    @inbounds for n in 1:nado
        # システム項: -i[H, ρₙ]
        @views mul!(dP[:, n], Ls, P[:, n], 1.0, 1.0)
        
        # 減衰項: -Σₖ nₖγₖ ρₙ
        @views dP[:, n] .-= ngamma[n] .* P[:, n]
        
        if nbath == 1
            # 単一熱浴
            ibath = 1
            nterms2 = nterms ÷ 2
            
            # phi 項（前進接続）
            PTMPx = zeros(ComplexF64, NL2)
            for j in 1:nterms
                np = idx_plus[j, n]
                if np > 0
                    @views PTMPx .+= phi[j, n] .* P[:, np]
                end
            end
            @views mul!(dP[:, n], Vx[ibath], PTMPx, -1.0im, 1.0)
            
            # theta 項（後退接続）
            PTMPl = zeros(ComplexF64, NL2)
            PTMPr = zeros(ComplexF64, NL2)
            for j in 1:nterms2
                nm = idx_minus[j, n]
                if nm > 0
                    @views PTMPl .+= theta_l[j, n] .* P[:, nm]
                end
            end
            for j in (nterms2 + 1):nterms
                nm = idx_minus[j, n]
                if nm > 0
                    @views PTMPr .+= theta_r[j, n] .* P[:, nm]
                end
            end
            @views mul!(dP[:, n], Vl[ibath], PTMPl, -1.0im, 1.0)
            @views mul!(dP[:, n], Vr[ibath], PTMPr, -1.0im, 1.0)
        else
            # 複数熱浴
            for ibath in 1:nbath
                jstart = noise.jstart_bath[ibath]
                nterms_b = noise.nterms_bath[ibath]
                jmid = jstart + nterms_b ÷ 2 - 1
                jend = jstart + nterms_b - 1
                
                # phi 項
                PTMPx = zeros(ComplexF64, NL2)
                for j in jstart:jend
                    np = idx_plus[j, n]
                    if np > 0
                        @views PTMPx .+= phi[j, n] .* P[:, np]
                    end
                end
                @views mul!(dP[:, n], Vx[ibath], PTMPx, -1.0im, 1.0)
                
                # theta 項
                PTMPl = zeros(ComplexF64, NL2)
                PTMPr = zeros(ComplexF64, NL2)
                for j in jstart:jmid
                    nm = idx_minus[j, n]
                    if nm > 0
                        @views PTMPl .+= theta_l[j, n] .* P[:, nm]
                    end
                end
                for j in (jmid + 1):jend
                    nm = idx_minus[j, n]
                    if nm > 0
                        @views PTMPr .+= theta_r[j, n] .* P[:, nm]
                    end
                end
                @views mul!(dP[:, n], Vl[ibath], PTMPl, -1.0im, 1.0)
                @views mul!(dP[:, n], Vr[ibath], PTMPr, -1.0im, 1.0)
            end
        end
    end
    
    return nothing
end

"""
    liouville(P::Matrix{ComplexF64}, system::HEOMSystem)

HEOM の Liouville 演算子を適用する（新しい配列を返す）。
"""
function liouville(P::Matrix{ComplexF64}, system::HEOMSystem)
    dP = similar(P)
    liouville!(dP, P, system)
    return dP
end
