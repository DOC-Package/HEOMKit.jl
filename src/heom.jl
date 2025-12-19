"""
    HEOM module

Define HEOM structures and Liouville operator.
"""

# =====================================
# HEOM operators structure
# =====================================

"""
    HEOMOperators

HEOM connection coefficients: `ngamma`, `phi`, `theta_l`, `theta_r`.
"""
struct HEOMOperators
    ngamma::Vector{ComplexF64}
    phi::Matrix{ComplexF64}
    theta_l::Matrix{ComplexF64}
    theta_r::Matrix{ComplexF64}
end

"""    HEOMOperators(noise, ado_idx, nado)

Construct HEOM operators from Noise and hierarchy indices.
"""
function HEOMOperators(noise::NoiseExp, ado_idx::Matrix{Int}, nado::Int)
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
    
    # theta_l, theta_r: separate first half / second half within each bath
    theta_l = zeros(ComplexF64, nterms, nado)
    theta_r = zeros(ComplexF64, nterms, nado)
    
    for n in 1:nado
        for ibath in 1:nbath
            jstart = noise.jstart_bath[ibath]
            nterms_b = noise.nterms_bath[ibath]
            jmid = jstart + nterms_b ÷ 2 - 1
            jend = jstart + nterms_b - 1
            
            # First half (original data)
            for j in jstart:jmid
                if ado_idx[j, n] > 0 && noise.abs_coeff[j] > 0
                    theta_l[j, n] = sqrt(ado_idx[j, n] / noise.abs_coeff[j]) * noise.coeff[j]
                end
            end
            # Second half (complex conjugate data)
            for j in (jmid + 1):jend
                if ado_idx[j, n] > 0 && noise.abs_coeff[j] > 0
                    theta_r[j, n] = -sqrt(ado_idx[j, n] / noise.abs_coeff[j]) * noise.coeff[j]
                end
            end
        end
    end
    
    return HEOMOperators(ngamma, phi, theta_l, theta_r)
end


# =====================================
# Complete HEOM system
# =====================================

"""
    HEOMSystem{M}

Complete HEOM system with all required data for time evolution.
Type parameter M determines the matrix type (SparseMat or DenseMat).

Fields: `noise`, `matrices`, `operators`, `ado_idx`, `idx_plus`, `idx_minus`, `nado`, `ndim`, `ndim2`
"""
struct HEOMSystem{M<:AbstractMatrix{ComplexF64}}
    noise::NoiseExp
    matrices::HEOMMatrices{M}
    operators::HEOMOperators
    ado_idx::Matrix{Int}
    idx_plus::Matrix{Int}
    idx_minus::Matrix{Int}
    nado::Int
    ndim::Int
    ndim2::Int
end

# Type aliases for convenience
const SparseHEOMSystem = HEOMSystem{SparseMatrixCSC{ComplexF64, Int}}
const DenseHEOMSystem = HEOMSystem{Matrix{ComplexF64}}

"""    HEOMSystem(H, noise, ndepth; hierarchy=:depth, sparse=true)

Construct HEOM system. 

# Arguments
- `H`: System Hamiltonian
- `noise`: Noise structure
- `ndepth`: Hierarchy depth
- `hierarchy`: Hierarchy construction method, `:depth` or `:width`
- `sparse`: If true (default), use sparse matrices. If false, use dense matrices.
"""
function HEOMSystem(H::AbstractMatrix, noise::NoiseExp, ndepth::Int;
                    hierarchy::Symbol=:depth, sparse::Bool=true)
    # Construct matrices
    matrices = HEOMMatrices(H, noise; sparse=sparse)
    
    # Construct hierarchy indices
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
    
    # Construct operators
    operators = HEOMOperators(noise, ado_idx, nado)
    
    return HEOMSystem(noise, matrices, operators, ado_idx, idx_plus, idx_minus,
                      nado, matrices.ndim, matrices.ndim2)
end


# =====================================
# Liouville operator
# =====================================

"""    liouville!(dP, P, system; parallel=false)

Apply HEOM Liouvillian to P (in-place).

# Arguments
- `parallel::Bool`: If true, use multi-threading for ADO loop.
"""
function liouville!(dP::Matrix{ComplexF64}, P::Matrix{ComplexF64}, system::HEOMSystem;
                    parallel::Bool=false)
    (; matrices, operators, noise, ado_idx, idx_plus, idx_minus, nado, ndim2) = system
    (; Ls, Vx, Vl, Vr) = matrices
    (; ngamma, phi, theta_l, theta_r) = operators
    nbath = noise.nbath
    
    dP .= 0.0 + 0.0im
    
    if parallel
        Threads.@threads for n in 1:nado
            _liouville_ado!(dP, P, n, Ls, Vx, Vl, Vr, ngamma, phi, theta_l, theta_r,
                           noise, idx_plus, idx_minus, ndim2)
        end
    else
        @inbounds for n in 1:nado
            _liouville_ado!(dP, P, n, Ls, Vx, Vl, Vr, ngamma, phi, theta_l, theta_r,
                           noise, idx_plus, idx_minus, ndim2)
        end
    end
    
    return nothing
end

"""Internal function for single ADO Liouvillian computation."""
@inline function _liouville_ado!(dP, P, n, Ls, Vx, Vl, Vr, ngamma, phi, theta_l, theta_r,
                                 noise, idx_plus, idx_minus, ndim2)
    nbath = noise.nbath
    
    # System term: -i[H, ρₙ]
    @views mul!(dP[:, n], Ls, P[:, n], 1.0, 1.0)
    
    # Damping term: -Σₖ nₖγₖ ρₙ
    @views dP[:, n] .-= ngamma[n] .* P[:, n]
    
    # 各熱浴について処理
    @inbounds for ibath in 1:nbath
        jstart = noise.jstart_bath[ibath]
        nterms_b = noise.nterms_bath[ibath]
        jmid = jstart + nterms_b ÷ 2 - 1
        jend = jstart + nterms_b - 1
        
        # phi 項（前進接続）
        PTMPx = zeros(ComplexF64, ndim2)
        for j in jstart:jend
            np = idx_plus[j, n]
            if np > 0
                @views PTMPx .+= phi[j, n] .* P[:, np]
            end
        end
        @views mul!(dP[:, n], Vx[ibath], PTMPx, -1.0im, 1.0)
        
        # theta 項（後退接続）
        PTMPl = zeros(ComplexF64, ndim2)
        PTMPr = zeros(ComplexF64, ndim2)
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
    
    return nothing
end

"""    liouville(P, system) → dP

Apply HEOM Liouvillian (allocating).
"""
function liouville(P::Matrix{ComplexF64}, system::HEOMSystem)
    dP = similar(P)
    liouville!(dP, P, system)
    return dP
end

"""    initial_ado(system, rho0::Matrix)

Create initial ADO from density matrix.
"""
function initial_ado(system::HEOMSystem, rho0::Matrix{ComplexF64})
    ndim2 = system.ndim2
    nado = system.nado
    
    P0 = zeros(ComplexF64, ndim2, nado)
    P0[:, 1] = vec(rho0)
    
    return P0
end

"""    initial_ado(system, state::Int=1)

Create initial ADO for pure state |⟨state⟩⟨state|.
"""
function initial_ado(system::HEOMSystem, state::Int=1)
    ndim = system.ndim
    rho0 = zeros(ComplexF64, ndim, ndim)
    rho0[state, state] = 1.0
    return initial_ado(system, rho0)
end