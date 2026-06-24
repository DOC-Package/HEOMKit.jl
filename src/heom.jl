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
        ngamma[n] = sum(ado_idx[j, n] * noise.γ[j] for j in 1:nterms)
    end
    
    # phi: √((nₖ+1)|cₖ|)
    phi = Matrix{ComplexF64}(undef, nterms, nado)
    for n in 1:nado
        for j in 1:nterms
            phi[j, n] = sqrt((ado_idx[j, n] + 1) * noise.abs_coeff[j])
        end
    end
    
    # theta_l, theta_r: use c1 and c2 directly
    # theta_l corresponds to c1 (left action), theta_r to c2 (right action)
    theta_l = zeros(ComplexF64, nterms, nado)
    theta_r = zeros(ComplexF64, nterms, nado)
    
    for n in 1:nado
        for j in 1:nterms
            if ado_idx[j, n] > 0 && noise.abs_coeff[j] > 0
                factor = sqrt(ado_idx[j, n] / noise.abs_coeff[j])
                theta_l[j, n] = factor * noise.c1[j]
                theta_r[j, n] = -factor * noise.c2[j]
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

"""    HEOMSystem(H, noise, ndepth; hierarchy=:depth, sparse=true, tolerance=1e-6, filter=false)

Construct HEOM system. 

# Arguments
- `H`: System Hamiltonian
- `noise`: Noise structure
- `ndepth`: Hierarchy depth
- `hierarchy`: Hierarchy construction method, `:depth` or `:width`
- `sparse`: If true (default), use sparse matrices. If false, use dense matrices.
- `tolerance`: Filtering threshold used when `hierarchy=:width` and `filter=true`
- `filter`: Enable width-based filtering when `hierarchy=:width`
"""
function HEOMSystem(H::AbstractMatrix, noise::NoiseExp, ndepth::Int;
                    hierarchy::Symbol=:depth, sparse::Bool=true,
                    tolerance::Real=1e-6, filter::Bool=false)
    # Construct matrices
    matrices = HEOMMatrices(H, noise; sparse=sparse)
    
    # Construct hierarchy indices
    nterms = noise.nterms
    if hierarchy == :depth
        nado, ado_idx, idx_plus, idx_minus = hierarchy_index_depth(nterms, ndepth)
    elseif hierarchy == :width
        nado, ado_idx, idx_plus, idx_minus = hierarchy_index_width(
            noise, ndepth;
            tolerance=tolerance, filter=filter
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
    (; matrices, operators, noise, idx_plus, idx_minus, nado, ndim2) = system
    (; Ls, Vx, Vl, Vr) = matrices
    (; ngamma, phi, theta_l, theta_r) = operators
    nterms = noise.nterms
    
    dP .= 0.0 + 0.0im
    
    if parallel
        Threads.@threads for n in 1:nado
            _liouville_ado!(dP, P, n, Ls, Vx, Vl, Vr, ngamma, phi, theta_l, theta_r,
                           idx_plus, idx_minus, nterms, ndim2)
        end
    else
        @inbounds for n in 1:nado
            _liouville_ado!(dP, P, n, Ls, Vx, Vl, Vr, ngamma, phi, theta_l, theta_r,
                           idx_plus, idx_minus, nterms, ndim2)
        end
    end
    
    return nothing
end

"""Internal function for single ADO Liouvillian computation."""
@inline function _liouville_ado!(dP, P, n, Ls, Vx, Vl, Vr, ngamma, phi, theta_l, theta_r,
                                 idx_plus, idx_minus, nterms, ndim2)
    # System term: -i[H, ρₙ]
    @views mul!(dP[:, n], Ls, P[:, n], 1.0, 1.0)
    
    # Damping term: -Σₖ nₖγₖ ρₙ
    @views dP[:, n] .-= ngamma[n] .* P[:, n]
    
    # Mode-by-mode processing (no bath loop needed)
    @inbounds for j in 1:nterms
        # phi term (forward connection): -i √((nₖ+1)|cₖ|) [V, ρₙ₊₁ₖ]
        np = idx_plus[j, n]
        if np > 0
            @views mul!(dP[:, n], Vx[j], P[:, np], -1.0im * phi[j, n], 1.0)
        end
        
        # theta terms (backward connection)
        # theta_l: c1 contribution (left action Vρ)
        # theta_r: c2 contribution (right action ρV†)
        nm = idx_minus[j, n]
        if nm > 0
            @views mul!(dP[:, n], Vl[j], P[:, nm], -1.0im * theta_l[j, n], 1.0)
            @views mul!(dP[:, n], Vr[j], P[:, nm], -1.0im * theta_r[j, n], 1.0)
        end
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