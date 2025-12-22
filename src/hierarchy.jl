# =====================================
# Helper Functions (must be defined first for @generated)
# =====================================

"""
Precompute log-factorials for fast norm calculation.
"""
@inline function _precompute_log_factorials(max_n::Int)
    log_fact = zeros(Float64, max_n + 1)
    log_fact[1] = 0.0  # log(0!) = 0
    for i in 1:max_n
        log_fact[i + 1] = log_fact[i] + log(i)
    end
    return log_fact
end

"""
Fast norm check using precomputed log-factorials.
Returns true if norm > tolerance.
"""
@inline function _check_norm_fast(idx::AbstractVector{Int}, 
                                   log_a::AbstractVector{Float64},
                                   log_fact::AbstractVector{Float64},
                                   tolerance::Real)
    log_W = 0.0
    @inbounds for k in eachindex(idx)
        n_k = idx[k]
        if n_k > 0
            log_W += n_k * log_a[k] - 0.5 * log_fact[n_k + 1]
        end
    end
    return exp(log_W) > tolerance
end

"""
Convert vector to tuple for NTuple{K} (compile-time generated).
"""
@generated function _vec_to_tuple(::Val{K}, v::Vector{Int}) where K
    exprs = [:(v[$i]) for i in 1:K]
    return Expr(:tuple, exprs...)
end


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
            a = _compute_filtering_weights_hseom(c, S_norm)
        elseif γ !== nothing
            # HEOM: Use decay-rate-based weights
            γ_eff = Float64.(real.(γ))
            a = _compute_filtering_weights_heom(c, S_norm, γ_eff)
        else
            error("Either D (matrix) or γ (vector) required for filtering")
        end
        log_a = log.(a)  # Precompute for fast norm calculation
    else
        a = nothing
        log_a = nothing
    end
    
    # Use optimized implementation
    return _hierarchy_bfs_optimized(nmode, ndepth, a, log_a, tolerance, filter)
end


"""
Optimized BFS hierarchy construction using NTuple keys for fast hashing.
"""
function _hierarchy_bfs_optimized(nmode::Int, ndepth::Int, 
                                   a::Union{Nothing, Vector{Float64}},
                                   log_a::Union{Nothing, Vector{Float64}},
                                   tolerance::Real, filter::Bool)
    # Dispatch to generated function based on nmode
    if nmode <= 16
        return _hierarchy_bfs_impl(Val(nmode), ndepth, a, log_a, tolerance, filter)
    else
        # Fallback for large nmode: use Vector keys
        return _hierarchy_bfs_vector(nmode, ndepth, a, log_a, tolerance, filter)
    end
end


"""
Generated function for compile-time specialization on nmode.
Uses index-based queue iteration (O(1) access) instead of popfirst! (O(n)).
"""
@generated function _hierarchy_bfs_impl(::Val{K}, ndepth::Int,
                                         a::Union{Nothing, Vector{Float64}},
                                         log_a::Union{Nothing, Vector{Float64}},
                                         tolerance::Real, filter::Bool) where K
    # Generate the zero tuple at compile time
    zero_tuple = Expr(:tuple, [0 for _ in 1:K]...)
    
    quote
        nmode = $K
        IdxType = NTuple{$K, Int}
        
        # Estimate capacity
        estimated_size = min(binomial(ndepth + nmode, nmode), 10_000_000)
        idx_hm = Dict{IdxType, Int}()
        sizehint!(idx_hm, estimated_size)
        
        # Flat storage for indices only
        ado_data = Vector{Int}()
        sizehint!(ado_data, estimated_size * $K)
        
        # Queue: (parent_ado_number, k_min)
        queue = Vector{Tuple{Int, Int}}()
        sizehint!(queue, estimated_size)
        
        # Precompute log-factorials
        log_fact = filter ? _precompute_log_factorials(ndepth + 1) : Float64[]
        
        # Initialize with zero
        idx_zero::IdxType = $zero_tuple
        idx_hm[idx_zero] = 1
        for i in 1:$K
            push!(ado_data, 0)
        end
        push!(queue, (1, 1))
        
        # Working buffer
        idx_buf = zeros(Int, $K)
        nado = 1
        queue_head = 1
        
        # BFS
        while queue_head <= length(queue)
            parent_n, k_min = queue[queue_head]
            queue_head += 1
            
            # Read parent index
            parent_offset = (parent_n - 1) * $K
            @inbounds for k in 1:$K
                idx_buf[k] = ado_data[parent_offset + k]
            end
            
            for k in k_min:$K
                @inbounds idx_buf[k] += 1
                
                # Fast depth calculation
                depth = 0
                @inbounds for i in 1:$K
                    depth += idx_buf[i]
                end
                
                if depth <= ndepth
                    idx_tuple = _vec_to_tuple(Val($K), idx_buf)
                    
                    if !haskey(idx_hm, idx_tuple)
                        include_ado = true
                        if filter
                            include_ado = _check_norm_fast(idx_buf, log_a, log_fact, tolerance)
                        end
                        
                        if include_ado
                            nado += 1
                            idx_hm[idx_tuple] = nado
                            @inbounds for i in 1:$K
                                push!(ado_data, idx_buf[i])
                            end
                            push!(queue, (nado, k))
                        end
                    end
                end
                
                @inbounds idx_buf[k] -= 1
            end
        end
        
        # Convert to matrix
        ado_idx = zeros(Int, $K, nado)
        @inbounds for n in 1:nado
            offset = (n - 1) * $K
            for k in 1:$K
                ado_idx[k, n] = ado_data[offset + k]
            end
        end
        
        # Build connections
        idx_plus, idx_minus = _build_connections_impl(Val($K), ado_idx, idx_hm, nado)
        
        return nado, ado_idx, idx_plus, idx_minus
    end
end


"""
Build connections with generated function.
"""
@generated function _build_connections_impl(::Val{K}, ado_idx::Matrix{Int},
                                             idx_hm::Dict, nado::Int) where K
    quote
        idx_plus, idx_minus = _build_connections_parallel(Val($K), ado_idx, idx_hm, nado)
        return idx_plus, idx_minus
    end
end

"""
Build connections with optional parallelization.
Separated from @generated to allow Threads.@threads.
"""
function _build_connections_parallel(::Val{K}, ado_idx::Matrix{Int},
                                      idx_hm::Dict, nado::Int) where K
    idx_plus = fill(-1, K, nado)
    idx_minus = fill(-1, K, nado)
    
    # Parallelize if large enough
    if nado > 10000 && Threads.nthreads() > 1
        Threads.@threads for n in 1:nado
            idx_buf = zeros(Int, K)  # Thread-local buffer
            @inbounds for k in 1:K
                idx_buf[k] = ado_idx[k, n]
            end
            
            @inbounds for k in 1:K
                # Forward
                idx_buf[k] += 1
                idx_tuple = _vec_to_tuple(Val(K), idx_buf)
                idx_plus[k, n] = get(idx_hm, idx_tuple, -1)
                idx_buf[k] -= 1
                
                # Backward
                if idx_buf[k] > 0
                    idx_buf[k] -= 1
                    idx_tuple = _vec_to_tuple(Val(K), idx_buf)
                    idx_minus[k, n] = get(idx_hm, idx_tuple, -1)
                    idx_buf[k] += 1
                end
            end
        end
    else
        idx_buf = zeros(Int, K)
        @inbounds for n in 1:nado
            for k in 1:K
                idx_buf[k] = ado_idx[k, n]
            end
            
            for k in 1:K
                # Forward
                idx_buf[k] += 1
                idx_tuple = _vec_to_tuple(Val(K), idx_buf)
                idx_plus[k, n] = get(idx_hm, idx_tuple, -1)
                idx_buf[k] -= 1
                
                # Backward
                if idx_buf[k] > 0
                    idx_buf[k] -= 1
                    idx_tuple = _vec_to_tuple(Val(K), idx_buf)
                    idx_minus[k, n] = get(idx_hm, idx_tuple, -1)
                    idx_buf[k] += 1
                end
            end
        end
    end
    
    return idx_plus, idx_minus
end



"""
Fallback for nmode > 16 using Vector keys (slower but works for any size).
Uses index-based queue iteration for O(1) access.
"""
function _hierarchy_bfs_vector(nmode::Int, ndepth::Int,
                                a::Union{Nothing, Vector{Float64}},
                                log_a::Union{Nothing, Vector{Float64}},
                                tolerance::Real, filter::Bool)
    estimated_size = min(binomial(ndepth + nmode, nmode), 10_000_000)
    
    # Use UInt64 hash as key for large nmode
    idx_hm = Dict{UInt64, Int}()
    sizehint!(idx_hm, estimated_size)
    
    # Flat storage: [idx1..., idx_nmode, k_min]
    stride = nmode + 1
    ado_data = Vector{Int}()
    sizehint!(ado_data, estimated_size * stride)
    
    log_fact = filter ? _precompute_log_factorials(ndepth + 1) : Float64[]
    
    # Initialize
    idx_zero = zeros(Int, nmode)
    h = _compute_hash(idx_zero)
    idx_hm[h] = 1
    append!(ado_data, idx_zero)
    push!(ado_data, 1)  # k_min
    
    idx_buf = zeros(Int, nmode)
    nado = 1
    queue_head = 1
    
    while queue_head <= nado
        offset = (queue_head - 1) * stride
        
        @inbounds for k in 1:nmode
            idx_buf[k] = ado_data[offset + k]
        end
        @inbounds k_min = ado_data[offset + nmode + 1]
        
        for k in k_min:nmode
            @inbounds idx_buf[k] += 1
            
            depth = 0
            @inbounds for i in 1:nmode
                depth += idx_buf[i]
            end
            
            if depth <= ndepth
                h = _compute_hash(idx_buf)
                
                if !haskey(idx_hm, h)
                    include_ado = true
                    if filter
                        include_ado = _check_norm_fast(idx_buf, log_a, log_fact, tolerance)
                    end
                    
                    if include_ado
                        nado += 1
                        idx_hm[h] = nado
                        append!(ado_data, idx_buf)
                        push!(ado_data, k)  # k_min
                    end
                end
            end
            
            @inbounds idx_buf[k] -= 1
        end
        
        queue_head += 1
    end
    
    # Convert to matrix
    ado_idx = zeros(Int, nmode, nado)
    @inbounds for n in 1:nado
        offset = (n - 1) * stride
        for k in 1:nmode
            ado_idx[k, n] = ado_data[offset + k]
        end
    end
    
    # Rebuild hash map with matrix data (for connection building)
    empty!(idx_hm)
    @inbounds for n in 1:nado
        h = _compute_hash(view(ado_idx, :, n))
        idx_hm[h] = n
    end
    
    # Build connections
    idx_plus = fill(-1, nmode, nado)
    idx_minus = fill(-1, nmode, nado)
    idx_buf2 = zeros(Int, nmode)
    
    @inbounds for n in 1:nado
        for k in 1:nmode
            idx_buf2[k] = ado_idx[k, n]
        end
        
        for k in 1:nmode
            idx_buf2[k] += 1
            h = _compute_hash(idx_buf2)
            idx_plus[k, n] = get(idx_hm, h, -1)
            idx_buf2[k] -= 1
            
            if idx_buf2[k] > 0
                idx_buf2[k] -= 1
                h = _compute_hash(idx_buf2)
                idx_minus[k, n] = get(idx_hm, h, -1)
                idx_buf2[k] += 1
            end
        end
    end
    
    return nado, ado_idx, idx_plus, idx_minus
end


"""
Compute hash for index vector (for large nmode fallback).
"""
@inline function _compute_hash(idx::AbstractVector{Int})
    h = UInt64(0)
    @inbounds for i in eachindex(idx)
        h = hash(idx[i], h)
    end
    return h
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
