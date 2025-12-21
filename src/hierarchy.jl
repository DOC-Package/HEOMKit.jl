"""
    hierarchy_index_depth(nmode, ndepth)

Return (nado, ado_idx, idx_plus, idx_minus) where each mode
index is now 1-based instead of 0-based.

# Arguments
- `nmode::Int`: Number of noise modes (Jmax)
- `ndepth::Int`: Maximum hierarchy depth (Nmax)

# Returns
- `nado::Int`: Total number of ADOs
- `ado_idx::Matrix{Int}`: Index vectors for each ADO (nmode × nado)
- `idx_plus::Matrix{Int}`: Forward (+1) connection indices
- `idx_minus::Matrix{Int}`: Backward (-1) connection indices
"""
function hierarchy_index_depth(nmode::Int, ndepth::Int)
    nado = binomial(ndepth + nmode, nmode)
    ado_idx = zeros(Int, nmode, nado)
    idx_plus = zeros(Int, nmode, nado)
    idx_minus = zeros(Int, nmode, nado)

    jk_temp = zeros(Int, nmode)
    hierarchy_index_depth_sub!(nmode, ndepth, jk_temp, 0, 1,
                         ado_idx, idx_plus, idx_minus)

    return nado, ado_idx, idx_plus, idx_minus
end


function hierarchy_index_depth_sub!(nmode::Int, ndepth::Int,
                              jk_temp::Vector{Int},
                              n::Int, k::Int,
                              ado_idx::Array{Int,2},
                              idx_plus::Array{Int,2},
                              idx_minus::Array{Int,2})
    jk_save = copy(jk_temp)
    for n_prime in n:ndepth
        if k < nmode
            hierarchy_index_depth_sub!(nmode, ndepth, jk_temp,
                                 n_prime, k + 1,
                                 ado_idx, idx_plus, idx_minus)
        else
            index = calc_index(nmode, jk_temp, n_prime) + 1
            ado_idx[:, index] = jk_temp

            # 1 step forward
            if n_prime < ndepth
                for k_prime in 1:nmode
                    jk_p1 = copy(jk_temp)
                    jk_p1[k_prime] += 1
                    idx_plus[k_prime, index] =
                        calc_index(nmode, jk_p1, n_prime + 1) + 1
                end
            else
                idx_plus[:, index] .= -1
            end

            # 1 step backward
            for k_prime in 1:nmode
                jk_m1 = copy(jk_temp)
                jk_m1[k_prime] -= 1
                idx_minus[k_prime, index] =
                    jk_m1[k_prime] ≥ 0 ? 
                    calc_index(nmode, jk_m1, n_prime - 1) + 1 : -1
            end
        end
        jk_temp[k] += 1
    end

    jk_temp .= jk_save
    return nothing
end


function calc_index(nmode::Int, jk_target::Vector{Int}, n::Int)
    res = binomial(n + nmode - 1, nmode)
    for k in 1:nmode
        s = sum(jk_target[1:k])
        for i in (s + 1):n
            res += binomial(n - i + nmode - k - 1, nmode - k - 1)
        end
    end
    return res
end


"""
    hierarchy_index_width(nmode, ndepth; D=nothing, γ=nothing, c=nothing, S_norm=1.0, 
                         tolerance=1e-6, filter=false)

Return (nado, ado_idx, idx_plus, idx_minus) using BFS-based hierarchy construction.
Supports norm-based filtering for HSEOM and HEOM.

# Filtering Criterion (HSEOM)
Effective decay rate: γₖᵉᶠᶠ = -(Re(Dₖₖ) + Σₗ≠ₖ |Dₖₗ|)
Weight coefficient: aₖ = √|cₖ| × ‖S‖_max / γₖᵉᶠᶠ  
Norm: Wₙ = ∏ₖ aₖⁿₖ / √(nₖ!)
Include ADO if: Wₙ > tolerance

# Arguments
- `nmode::Int`: Number of modes K
- `ndepth::Int`: Maximum hierarchy depth (safety limit)

# Keyword Arguments
- `D::Matrix`: Mode coupling matrix (K × K) for HSEOM
- `γ::Vector`: Decay rates for HEOM (alternative to D)
- `c::Vector`: BCF coefficients cₖ
- `S_norm::Real`: System operator norm ‖S‖_max (default: 1.0)
- `tolerance::Real`: Filtering threshold (default: 1e-6)
- `filter::Bool`: Enable filtering (default: false)

# Returns
- `nado::Int`: Total number of ADOs
- `ado_idx::Matrix{Int}`: Index vectors for each ADO
- `idx_plus::Matrix{Int}`: Forward connection indices  
- `idx_minus::Matrix{Int}`: Backward connection indices

# Examples
```julia
# HEOM (diagonal decay)
γ = [0.5, 1.0, 2.0]
c = [1.0, 0.5, 0.3]
nado, ado_idx, idx_plus, idx_minus = hierarchy_index_width(3, 10; 
    γ=γ, c=c, tolerance=1e-5, filter=true)

# HSEOM (non-diagonal D)
D = [-1.0 0.2; 0.2 -2.0]
c = [1.0, 0.5]
nado, ado_idx, idx_plus, idx_minus = hierarchy_index_width(2, 10;
    D=D, c=c, S_norm=1.0, tolerance=1e-5, filter=true)
```
"""
function hierarchy_index_width(nmode::Int, ndepth::Int;
                               D::Union{Nothing, AbstractMatrix}=nothing,
                               γ::Union{Nothing, AbstractVector}=nothing,
                               c::Union{Nothing, AbstractVector}=nothing,
                               S_norm::Real=1.0,
                               tolerance::Real=1e-6,
                               filter::Bool=false)
    
    # Compute filtering weights if filtering enabled
    if filter
        if c === nothing
            error("c (BCF coefficients) required for filtering")
        end
        
        # Compute filtering weights based on method
        if D !== nothing
            # HSEOM: Use coefficient-based weights (no decay rate division)
            # aₖ = √|cₖ| × ‖S‖_max
            # This is appropriate for oscillatory bases like PSWF where D_kk is purely imaginary
            a = _compute_filtering_weights_hseom(c, S_norm)
        elseif γ !== nothing
            # HEOM: Use decay-rate-based weights
            # aₖ = √|cₖ| × ‖S‖_max / γₖ
            γ_eff = Float64.(real.(γ))
            a = _compute_filtering_weights_heom(c, S_norm, γ_eff)
        else
            error("Either D (matrix) or γ (vector) required for filtering")
        end
    else
        a = nothing
    end
    
    # BFS-based hierarchy construction
    idx_hm = Dict{Vector{Int}, Int}()
    ado_list = Vector{Vector{Int}}()
    queue = Vector{Tuple{Vector{Int}, Int}}()
    
    # Initialize with zero vector
    idx_zero = zeros(Int, nmode)
    push!(queue, (idx_zero, 1))
    idx_hm[copy(idx_zero)] = 1
    push!(ado_list, copy(idx_zero))
    
    while !isempty(queue)
        idx, k_min = popfirst!(queue)
        
        for k in k_min:nmode
            idx_child = copy(idx)
            idx_child[k] += 1
            depth = sum(idx_child)
            
            # Skip if already visited
            if haskey(idx_hm, idx_child)
                continue
            end
            
            # Depth limit
            if depth > ndepth
                continue
            end
            
            # Filtering criterion
            include_ado = true
            if filter && a !== nothing
                W = _compute_hierarchy_norm(idx_child, a)
                include_ado = W > tolerance
            end
            
            if include_ado
                n_new = length(ado_list) + 1
                idx_hm[copy(idx_child)] = n_new
                push!(ado_list, copy(idx_child))
                push!(queue, (idx_child, k))
            end
        end
    end
    
    nado = length(ado_list)
    
    # Build arrays
    ado_idx = zeros(Int, nmode, nado)
    for n in 1:nado
        ado_idx[:, n] = ado_list[n]
    end
    
    # Build connection matrices
    idx_plus = fill(-1, nmode, nado)
    idx_minus = fill(-1, nmode, nado)
    
    for n in 1:nado
        idx = ado_list[n]
        for k in 1:nmode
            # Forward
            idx_p = copy(idx)
            idx_p[k] += 1
            idx_plus[k, n] = get(idx_hm, idx_p, -1)
            
            # Backward
            if idx[k] > 0
                idx_m = copy(idx)
                idx_m[k] -= 1
                idx_minus[k, n] = get(idx_hm, idx_m, -1)
            end
        end
    end
    
    return nado, ado_idx, idx_plus, idx_minus
end

# Legacy interface for backward compatibility
function hierarchy_index_width(nmode::Int, ndepth::Int,
                               gamk::Vector, ck::Vector, bk::Vector;
                               ak::Real=1.0, tolerance::Real=1e-6,
                               filter::Bool=false)
    if filter
        # Use new filtering with γ = gamk
        return hierarchy_index_width(nmode, ndepth;
                                     γ=gamk, c=ck, tolerance=tolerance, filter=true)
    else
        # No filtering: just use depth limit
        return hierarchy_index_width(nmode, ndepth; filter=false)
    end
end


"""
    hierarchy_index_width(noise::NoiseExp, ndepth; S_norm=1.0, tolerance=1e-6, filter=false)

Construct hierarchy from NoiseExp (HEOM exponential expansion).
Extracts γ = real(expon) and c = coeff from the noise object.

# Arguments
- `noise::NoiseExp`: Noise parameters with exponential expansion
- `ndepth::Int`: Maximum hierarchy depth

# Keyword Arguments  
- `S_norm::Real`: System operator norm (default: 1.0, or computed from noise.V)
- `tolerance::Real`: Filtering threshold (default: 1e-6)
- `filter::Bool`: Enable filtering (default: false)
"""
function hierarchy_index_width(noise::NoiseExp, ndepth::Int;
                               S_norm::Union{Real, Nothing}=nothing,
                               tolerance::Real=1e-6,
                               filter::Bool=false)
    nmode = noise.nterms
    
    if filter
        # Extract parameters for filtering
        γ = real.(noise.expon)  # Decay rates
        c = noise.coeff         # BCF coefficients
        
        # Compute S_norm from interaction operators if not provided
        if S_norm === nothing
            S_norm = maximum(opnorm(V) for V in noise.V)
        end
        
        return hierarchy_index_width(nmode, ndepth; γ=γ, c=c, S_norm=S_norm, 
                                     tolerance=tolerance, filter=true)
    else
        return hierarchy_index_width(nmode, ndepth; filter=false)
    end
end


"""
    hierarchy_index_width(noise::NoiseGeneral, ndepth; S_norm=1.0, tolerance=1e-6, filter=false)

Construct hierarchy from NoiseGeneral (HSEOM general expansion).
Extracts D matrix and c = coeff from the noise object.

# Arguments
- `noise::NoiseGeneral`: Noise parameters with general basis expansion
- `ndepth::Int`: Maximum hierarchy depth

# Keyword Arguments
- `S_norm::Real`: System operator norm (default: 1.0, or computed from noise.V)
- `tolerance::Real`: Filtering threshold (default: 1e-6)
- `filter::Bool`: Enable filtering (default: false)
"""
function hierarchy_index_width(noise::NoiseGeneral, ndepth::Int;
                               S_norm::Union{Real, Nothing}=nothing,
                               tolerance::Real=1e-6,
                               filter::Bool=false)
    nmode = noise.nterms
    
    if filter
        # Extract parameters for filtering
        D = noise.D       # Derivative matrix
        c = noise.coeff   # BCF coefficients
        
        # Compute S_norm from interaction operators if not provided
        if S_norm === nothing
            S_norm = maximum(opnorm(V) for V in noise.V)
        end
        
        return hierarchy_index_width(nmode, ndepth; D=D, c=c, S_norm=S_norm,
                                     tolerance=tolerance, filter=true)
    else
        return hierarchy_index_width(nmode, ndepth; filter=false)
    end
end


# =====================================
# Internal Helper Functions
# =====================================

"""
Compute filtering weights for HEOM: aₖ = √|cₖ| × ‖S‖_max / γₖ

Uses decay rate γₖ in denominator (appropriate for exponential basis).
"""
function _compute_filtering_weights_heom(c::AbstractVector, S_norm::Real, γ::AbstractVector)
    K = length(c)
    a = zeros(Float64, K)
    for k in 1:K
        if γ[k] > 0
            a[k] = sqrt(abs(c[k])) * S_norm / γ[k]
        else
            a[k] = Inf  # Include non-decaying modes
        end
    end
    return a
end

"""
Compute filtering weights for HSEOM: aₖ = √|cₖ| × ‖S‖_max

No decay rate division (appropriate for oscillatory bases like PSWF
where D_kk is purely imaginary and doesn't represent decay).
"""
function _compute_filtering_weights_hseom(c::AbstractVector, S_norm::Real)
    K = length(c)
    a = zeros(Float64, K)
    for k in 1:K
        a[k] = sqrt(abs(c[k])) * S_norm
    end
    return a
end

"""
Compute hierarchy norm: Wₙ = ∏ₖ aₖⁿₖ / √(nₖ!)
"""
function _compute_hierarchy_norm(n::AbstractVector{<:Integer}, a::AbstractVector{<:Real})
    log_W = 0.0
    for k in eachindex(n)
        if n[k] > 0
            if isinf(a[k])
                return Inf
            end
            log_W += n[k] * log(a[k]) - 0.5 * _logfactorial(n[k])
        end
    end
    return exp(log_W)
end

function _logfactorial(n::Integer)
    n <= 1 && return 0.0
    n <= 20 && return log(factorial(big(n)))
    return n * log(n) - n + 0.5 * log(2π * n)  # Stirling
end


# =====================================
# Exported utility functions for users
# =====================================


"""
    build_hseom_index_maps(ado_idx::Matrix{Int}, nado::Int, nmode::Int)

Construct general two-mode simultaneous change index maps for HSEOM (all k, ℓ pairs).

BCF expansion: C(t) = Σₖ cₖ φₖ(t), ∂ₜφₖ = Σₗ Dₖₗ φₗ(t)

Supports general D matrices (including non-tridiagonal).
Uses dictionary for O(1) fast access.

# Arguments
- `ado_idx::Matrix{Int}`: Hierarchy indices (nmode × nado)
- `nado::Int`: Total number of ADOs/ADWs
- `nmode::Int`: Number of modes

# Returns
NamedTuple with:
- `idx_minus_plus::Array{Int,3}`: (k, ℓ, n) → connection index for n[k]-1, n[ℓ]+1
- `idx_plus_minus::Array{Int,3}`: (k, ℓ, n) → connection index for n[k]+1, n[ℓ]-1

Array size is (nmode × nmode × nado).
Diagonal entries (k == ℓ) are -1 (single-mode changes use existing idx_minus, idx_plus).
Non-existent connections are also -1.

# Example
```julia
nado, ado_idx, idx_plus, idx_minus = hierarchy_index_depth(3, 4)
hseom_idx = build_hseom_index_maps(ado_idx, nado, 3)
# For ADO n=5 with modes k=2, ℓ=1
n_mp = hseom_idx.idx_minus_plus[2, 1, 5]  # n₂-1, n₁+1
n_pm = hseom_idx.idx_plus_minus[2, 1, 5]  # n₂+1, n₁-1
```
"""
function build_hseom_index_maps(ado_idx::Matrix{Int}, nado::Int, nmode::Int)
    # Dictionary: ADO index vector → ADO number
    idx_to_ado = Dict{Vector{Int}, Int}()
    for n in 1:nado
        idx_to_ado[ado_idx[:, n]] = n
    end
    
    # Initialize result 3D arrays
    # idx_minus_plus[k, ℓ, n]: connection to n[k]-1, n[ℓ]+1
    # idx_plus_minus[k, ℓ, n]: connection to n[k]+1, n[ℓ]-1
    idx_minus_plus = fill(-1, nmode, nmode, nado)
    idx_plus_minus = fill(-1, nmode, nmode, nado)
    
    # Compute all (k, ℓ) pairs for each ADO
    for n in 1:nado
        idx = copy(ado_idx[:, n])
        
        for k in 1:nmode
            for ℓ in 1:nmode
                if k == ℓ
                    continue  # Diagonal entries are single-mode changes (handled by existing idx_minus, idx_plus)
                end
                
                # idx_minus_plus: n[k] - 1, n[ℓ] + 1
                idx_tmp = copy(idx)
                idx_tmp[k] -= 1
                idx_tmp[ℓ] += 1
                if idx_tmp[k] >= 0
                    idx_minus_plus[k, ℓ, n] = get(idx_to_ado, idx_tmp, -1)
                end
                
                # idx_plus_minus: n[k] + 1, n[ℓ] - 1
                idx_tmp = copy(idx)
                idx_tmp[k] += 1
                idx_tmp[ℓ] -= 1
                if idx_tmp[ℓ] >= 0
                    idx_plus_minus[k, ℓ, n] = get(idx_to_ado, idx_tmp, -1)
                end
            end
        end
    end
    
    return (
        idx_minus_plus = idx_minus_plus,
        idx_plus_minus = idx_plus_minus
    )
end
