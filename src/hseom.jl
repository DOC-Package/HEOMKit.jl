"""
    HSEOM module

HSEOM (Hierarchical Stochastic Equations of Motion) の構造体と Liouville 演算子を定義する。
2モード同時変化を用いた確率的手法に対応。
Fortran の Liouville_ket/Liouville_bra に対応（規格化なし版）。

HSEOMは波動関数（ADW: Auxiliary Density Wavefunctions）を扱うため、
次元は ndim（ヒルベルト空間）である。
"""

# =====================================
# HSEOM 演算子構造体
# =====================================

"""
    HSEOMOperators

HSEOM 演算子構造体（規格化なし版）。
係数 nvec(j,n) を直接使用するため、adw_idx のみ保持。

# Fields
- `adw_idx::Matrix{Int}`: 階層インデックス nvec (nterms × nadw)
"""
struct HSEOMOperators
    adw_idx::Matrix{Int}
end


# =====================================
# HSEOM システム
# =====================================

"""
    HSEOMSystem

HSEOM 完全システム。2モード同時変化のインデックスを含む。
波動関数（ADW）を扱うため、次元は ndim（ヒルベルト空間）。

# Fields
- `H::Matrix{ComplexF64}`: システムハミルトニアン
- `V::Vector{Matrix{ComplexF64}}`: 相互作用演算子（各熱浴）
- `noise::Noise`: ノイズパラメータ
- `D::Matrix{ComplexF64}`: BCF 展開の D 行列（∂ₜφₖ = Σₗ Dₖₗ φₗ）
- `operators::HSEOMOperators`: HSEOM 演算子
- `adw_idx::Matrix{Int}`: 階層インデックス
- `idx_plus::Matrix{Int}`: 単一モード前進インデックス
- `idx_minus::Matrix{Int}`: 単一モード後退インデックス
- `idx_minus_plus::Array{Int,3}`: (k,ℓ,n) → n[k]-1, n[ℓ]+1 インデックス
- `idx_plus_minus::Array{Int,3}`: (k,ℓ,n) → n[k]+1, n[ℓ]-1 インデックス
- `nadw::Int`: ADW 総数
- `ndim::Int`: ヒルベルト空間次元
- `nterms::Int`: BCF 展開項数
"""
struct HSEOMSystem
    H::Matrix{ComplexF64}
    V::Vector{Matrix{ComplexF64}}
    noise::Noise
    D::Matrix{ComplexF64}
    operators::HSEOMOperators
    adw_idx::Matrix{Int}
    idx_plus::Matrix{Int}
    idx_minus::Matrix{Int}
    idx_minus_plus::Array{Int,3}
    idx_plus_minus::Array{Int,3}
    nadw::Int
    ndim::Int
    nterms::Int
end

"""
    HSEOMSystem(H::AbstractMatrix, noise::Noise, D::AbstractMatrix, ndepth::Int;
                hierarchy::Symbol=:depth)

HSEOM システムを構築する。

# Arguments
- `H::AbstractMatrix`: システムハミルトニアン
- `noise::Noise`: ノイズパラメータ
- `D::AbstractMatrix`: BCF 展開の D 行列（∂ₜφₖ = Σₗ Dₖₗ φₗ）
- `ndepth::Int`: 階層の深さ
- `hierarchy::Symbol`: 階層構築方法 (:depth または :width)

# Example
```julia
H = [0 100; 100 0]
bath = Bath(expon, coeff, V)
noise = Noise(bath)
# 三重対角 D 行列の例（Bessel展開）
D = build_tridiagonal_D(nterms, gamma_c)
system = HSEOMSystem(H, noise, D, 5)
```
"""
function HSEOMSystem(H::AbstractMatrix, noise::Noise, D::AbstractMatrix, ndepth::Int;
                     hierarchy::Symbol=:depth)
    ndim = size(H, 1)
    H_complex = Matrix{ComplexF64}(H)
    D_complex = Matrix{ComplexF64}(D)
    
    # 相互作用演算子を取得
    V = [Matrix{ComplexF64}(noise.V[ibath]) for ibath in 1:noise.nbath]
    
    # 階層インデックスを構築
    nterms = noise.nterms
    @assert size(D, 1) == nterms && size(D, 2) == nterms "D matrix size must be ($nterms, $nterms)"
    
    if hierarchy == :depth
        nadw, adw_idx, idx_plus, idx_minus = hierarchy_index_depth(nterms, ndepth)
    elseif hierarchy == :width
        bk, ak, _ = compute_heom_params(noise)
        nadw, adw_idx, idx_plus, idx_minus = hierarchy_index_width(
            nterms, ndepth, noise.expon, noise.coeff, bk;
            ak=ak, filter=false
        )
    else
        error("Unknown hierarchy method: $hierarchy. Use :depth or :width")
    end
    
    # 2モード変化インデックスを構築（一般化版）
    hseom_idx = build_hseom_index_maps(adw_idx, nadw, nterms)
    
    # 演算子を構築（規格化なし版は adw_idx のみ保持）
    operators = HSEOMOperators(adw_idx)
    
    return HSEOMSystem(
        H_complex, V, noise, D_complex, operators, adw_idx, idx_plus, idx_minus,
        hseom_idx.idx_minus_plus,
        hseom_idx.idx_plus_minus,
        nadw, ndim, nterms
    )
end


# =====================================
# D 行列構築ヘルパー関数
# =====================================

"""
    build_tridiagonal_D(nterms::Int, gamma_c::Number)

三重対角 D 行列を構築する（Bessel展開用）。

D[k,k] = 0（対角成分は単一モード変化で処理）
D[k,k-1] = +γc/2
D[k,k+1] = -γc/2
ただし k=1 は D[1,2] = -γc
"""
function build_tridiagonal_D(nterms::Int, gamma_c::Number)
    D = zeros(ComplexF64, nterms, nterms)
    gc = ComplexF64(gamma_c)
    
    # k=1: 特別な処理
    if nterms > 1
        D[1, 2] = -gc
    end
    
    # k=2 to nterms-1: 中間モード
    for k in 2:(nterms-1)
        D[k, k-1] = 0.5 * gc
        D[k, k+1] = -0.5 * gc
    end
    
    # k=nterms: 最後のモード（k+1が無い）
    if nterms > 1
        D[nterms, nterms-1] = 0.5 * gc
    end
    
    return D
end


# =====================================
# Liouville 演算子（一般 D 行列対応版）
# =====================================

"""
    liouville_ket!(dP::Matrix{ComplexF64}, P::Matrix{ComplexF64}, system::HSEOMSystem)

HSEOM の ket 側 Liouville 演算子を適用（in-place）。
Fortran の Liouville_ket に完全対応。

Fortran コード:
```
P1(:,n) = - iconst*smul(Hs,P(:,n))
P1(:,n) = P1(:,n) - dble(nvec(0,n))*gamc*P(:,nmpv2(0,n))
do j = 1, Jmax-1
    P1(:,n) = P1(:,n) + 0.5d0*dble(nvec(j,n))*gamc*(P(:,nmpv1(j,n))-P(:,nmpv2(j,n)))
end do
PTMP(:) = dble(nvec(0,n))*bessel_jn(0,0d0)*P(:,nmvec(0,n))
do j = 0, Jmax-1
    PTMP(:) = PTMP(:) + ck(j)*P(:,npvec(j,n)) 
end do
P1(:,n) = P1(:,n) - iconst*smul(V,PTMP) 
```

nmpv1(k,n) = n[k]-1, n[k-1]+1 = idx_minus_plus[k, k-1, n]
nmpv2(k,n) = n[k]-1, n[k+1]+1 = idx_minus_plus[k, k+1, n]
"""
function liouville_ket!(dP::Matrix{ComplexF64}, P::Matrix{ComplexF64}, system::HSEOMSystem)
    (; H, V, D, noise, adw_idx, nadw, ndim, nterms) = system
    (; idx_plus, idx_minus, idx_minus_plus) = system
    
    nbath = noise.nbath
    
    # D 行列から γc を取得（D[1,2] = -γc なので）
    gamma_c = -real(D[1, 2])
    
    dP .= 0.0 + 0.0im
    
    @inbounds for n in 1:nadw
        # システム項: -i * H * P(:,n)
        @views mul!(dP[:, n], H, P[:, n], -1.0im, 1.0)
        
        # 各熱浴について D 行列項を処理
        for ibath in 1:nbath
            jstart = noise.jstart_bath[ibath]
            nterms_b = noise.nterms_bath[ibath]
            jend = jstart + nterms_b - 1
            
            # j=0 (k=jstart): -nvec(0,n) * gamc * P(:, nmpv2(0,n))
            # nmpv2(0,n) = n[0]-1, n[1]+1 = idx_minus_plus[jstart, jstart+1, n]
            k = jstart
            nk = adw_idx[k, n]
            if nk > 0 && k + 1 <= nterms
                nm_pl = idx_minus_plus[k, k+1, n]
                if nm_pl > 0
                    coef = -Float64(nk) * gamma_c
                    @views dP[:, n] .+= coef .* P[:, nm_pl]
                end
            end
            
            # j=1 to Jmax-1 (k=jstart+1 to jend-1):
            # +0.5 * nvec(j,n) * gamc * (P(:,nmpv1(j,n)) - P(:,nmpv2(j,n)))
            for k in (jstart+1):(jend-1)
                nk = adw_idx[k, n]
                if nk == 0
                    continue
                end
                
                coef = 0.5 * Float64(nk) * gamma_c
                
                # nmpv1(k,n) = n[k]-1, n[k-1]+1
                nm_pl_prev = idx_minus_plus[k, k-1, n]
                if nm_pl_prev > 0
                    @views dP[:, n] .+= coef .* P[:, nm_pl_prev]
                end
                
                # nmpv2(k,n) = n[k]-1, n[k+1]+1
                nm_pl_next = idx_minus_plus[k, k+1, n]
                if nm_pl_next > 0
                    @views dP[:, n] .-= coef .* P[:, nm_pl_next]
                end
            end
            
            # 相互作用項
            PTMPx = zeros(ComplexF64, ndim)
            
            # 後退接続項: nvec(0,n) * bessel_jn(0,0) * P(:,nmvec(0,n))
            # bessel_jn(0,0) = 1.0
            k0 = jstart
            nk0 = adw_idx[k0, n]
            if nk0 > 0
                nm = idx_minus[k0, n]
                if nm > 0
                    @views PTMPx .+= Float64(nk0) .* P[:, nm]
                end
            end
            
            # 前進接続項: ck(j) * P(:,npvec(j,n)) for j=0 to Jmax-1
            for k in jstart:(jend-1)
                np = idx_plus[k, n]
                if np > 0
                    @views PTMPx .+= noise.coeff[k] .* P[:, np]
                end
            end
            
            # -i * V * PTMP
            @views mul!(dP[:, n], V[ibath], PTMPx, -1.0im, 1.0)
        end
    end
    
    return nothing
end

"""
    liouville_bra!(dP::Matrix{ComplexF64}, P::Matrix{ComplexF64}, system::HSEOMSystem)

HSEOM の bra 側 Liouville 演算子を適用（in-place）。
Fortran の Liouville_bra に完全対応。

Fortran コード:
```
P1(:,n) = - iconst*smul(Hs,P(:,n))
P1(:,n) = P1(:,n) + dble(nvec(0,n)+1)*gamc*P(:,npmv2(0,n))
do j = 1, Jmax-1
    P1(:,n) = P1(:,n) - 0.5d0*dble(nvec(j,n)+1)*gamc*(P(:,npmv1(j,n))-P(:,npmv2(j,n)))
end do
PTMP(:) = dble(nvec(0,n)+1)*bessel_jn(0,0d0)*P(:,npvec(0,n))
do j = 0, Jmax-1
    PTMP(:) = PTMP(:) + conjg(ck(j))*P(:,nmvec(j,n)) 
end do
P1(:,n) = P1(:,n) - iconst*smul(V,PTMP)
```

npmv1(k,n) = n[k]+1, n[k-1]-1 = idx_plus_minus[k, k-1, n]
npmv2(k,n) = n[k]+1, n[k+1]-1 = idx_plus_minus[k, k+1, n]
"""
function liouville_bra!(dP::Matrix{ComplexF64}, P::Matrix{ComplexF64}, system::HSEOMSystem)
    (; H, V, D, noise, adw_idx, nadw, ndim, nterms) = system
    (; idx_plus, idx_minus, idx_plus_minus) = system
    
    nbath = noise.nbath
    
    # D 行列から γc を取得（D[1,2] = -γc なので）
    gamma_c = -real(D[1, 2])
    
    dP .= 0.0 + 0.0im
    
    @inbounds for n in 1:nadw
        # システム項: -i * H * P(:,n)
        @views mul!(dP[:, n], H, P[:, n], -1.0im, 1.0)
        
        # 各熱浴について D† 行列項を処理
        for ibath in 1:nbath
            jstart = noise.jstart_bath[ibath]
            nterms_b = noise.nterms_bath[ibath]
            jend = jstart + nterms_b - 1
            
            # j=0 (k=jstart): +(nvec(0,n)+1) * gamc * P(:, npmv2(0,n))
            # npmv2(0,n) = n[0]+1, n[1]-1 = idx_plus_minus[jstart, jstart+1, n]
            k = jstart
            nk = adw_idx[k, n]
            if k + 1 <= nterms
                np_ml = idx_plus_minus[k, k+1, n]
                if np_ml > 0
                    coef = Float64(nk + 1) * gamma_c
                    @views dP[:, n] .+= coef .* P[:, np_ml]
                end
            end
            
            # j=1 to Jmax-1 (k=jstart+1 to jend-1):
            # -0.5 * (nvec(j,n)+1) * gamc * (P(:,npmv1(j,n)) - P(:,npmv2(j,n)))
            for k in (jstart+1):(jend-1)
                nk = adw_idx[k, n]
                
                coef = 0.5 * Float64(nk + 1) * gamma_c
                
                # npmv1(k,n) = n[k]+1, n[k-1]-1
                np_ml_prev = idx_plus_minus[k, k-1, n]
                if np_ml_prev > 0
                    @views dP[:, n] .-= coef .* P[:, np_ml_prev]
                end
                
                # npmv2(k,n) = n[k]+1, n[k+1]-1
                np_ml_next = idx_plus_minus[k, k+1, n]
                if np_ml_next > 0
                    @views dP[:, n] .+= coef .* P[:, np_ml_next]
                end
            end
            
            # 相互作用項
            PTMPx = zeros(ComplexF64, ndim)
            
            # 前進接続項: (nvec(0,n)+1) * bessel_jn(0,0) * P(:,npvec(0,n))
            # bessel_jn(0,0) = 1.0
            k0 = jstart
            nk0 = adw_idx[k0, n]
            np = idx_plus[k0, n]
            if np > 0
                @views PTMPx .+= Float64(nk0 + 1) .* P[:, np]
            end
            
            # 後退接続項: conjg(ck(j)) * P(:,nmvec(j,n)) for j=0 to Jmax-1
            for k in jstart:(jend-1)
                nm = idx_minus[k, n]
                if nm > 0
                    @views PTMPx .+= conj(noise.coeff[k]) .* P[:, nm]
                end
            end
            
            # -i * V * PTMP
            @views mul!(dP[:, n], V[ibath], PTMPx, -1.0im, 1.0)
        end
    end
    
    return nothing
end

"""
    liouville_ket(P::Matrix{ComplexF64}, system::HSEOMSystem)

HSEOM の ket 側 Liouville 演算子を適用（新しい配列を返す）。
"""
function liouville_ket(P::Matrix{ComplexF64}, system::HSEOMSystem)
    dP = similar(P)
    liouville_ket!(dP, P, system)
    return dP
end

"""
    liouville_bra(P::Matrix{ComplexF64}, system::HSEOMSystem)

HSEOM の bra 側 Liouville 演算子を適用（新しい配列を返す）。
"""
function liouville_bra(P::Matrix{ComplexF64}, system::HSEOMSystem)
    dP = similar(P)
    liouville_bra!(dP, P, system)
    return dP
end


# =====================================
# 初期条件
# =====================================

"""
    initial_adw(system::HSEOMSystem, psi0::Vector{ComplexF64})

初期 ADW を作成する。ψ₀ のみ非ゼロ、他の ADW はゼロ。
"""
function initial_adw(system::HSEOMSystem, psi0::Vector{ComplexF64})
    ndim = system.ndim
    nadw = system.nadw
    
    P0 = zeros(ComplexF64, ndim, nadw)
    P0[:, 1] = psi0
    
    return P0
end

"""
    initial_adw(system::HSEOMSystem, state::Int=1)

指定した状態 |state⟩ から始まる初期 ADW を作成する。
"""
function initial_adw(system::HSEOMSystem, state::Int=1)
    ndim = system.ndim
    psi0 = zeros(ComplexF64, ndim)
    psi0[state] = 1.0
    return initial_adw(system, psi0)
end
