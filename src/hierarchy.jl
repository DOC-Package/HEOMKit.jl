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
    hierarchy_index_width(nmode, ndepth, gamk, ck, bk; ak=1.0, tolerance=1e-6)

Return (nado, ado_idx, idx_plus, idx_minus) using breadth-first search (BFS) based hierarchy construction.
This method filters hierarchy indices based on a norm criterion.

# Arguments
- `nmode::Int`: Number of noise modes (Jmax)
- `ndepth::Int`: Maximum depth (Nmax)
- `gamk::Vector`: Gamma values for each mode
- `ck::Vector`: c coefficients for each mode
- `bk::Vector`: b coefficients for each mode
- `ak::Real`: Exponent parameter (default: 1.0)
- `tolerance::Real`: Tolerance for filtering (default: 1e-6)

# Returns
- `nado::Int`: Total number of ADOs
- `ado_idx::Matrix{Int}`: Index vectors for each ADO
- `idx_plus::Matrix{Int}`: Forward connection indices
- `idx_minus::Matrix{Int}`: Backward connection indices
"""
function hierarchy_index_width(nmode::Int, ndepth::Int,
                               gamk::Vector, ck::Vector, bk::Vector;
                               ak::Real=1.0, tolerance::Real=1e-6,
                               filter::Bool=false)
    # Hash map: index vector -> ADO index
    idx_hm = Dict{Vector{Int}, Int}()
    
    # Queues for BFS
    idx_queue = Vector{Vector{Int}}()
    klm_queue = Vector{Int}()
    
    # Estimate maximum number of indices
    n_est = binomial(ndepth + nmode, nmode)
    nvtmp = zeros(Int, nmode, n_est)
    
    # Initialize with zero vector
    idx = zeros(Int, nmode)
    push!(idx_queue, copy(idx))
    push!(klm_queue, 1)
    
    n_idx = 0  # 0-based counter for compatibility
    
    while !isempty(idx_queue)
        idx = popfirst!(idx_queue)
        klm = popfirst!(klm_queue)
        
        # Register current index
        idx_hm[copy(idx)] = n_idx
        nvtmp[:, n_idx + 1] = idx  # Julia is 1-based
        n_idx += 1
        
        # Explore neighbors
        for k in klm:nmode
            idx[k] += 1
            dpt = sum(idx)
            
            # Calculate norm for filtering
            ngam = sum(idx[j] * real(gamk[j]) for j in 1:nmode)
            
            norm_val = 1.0
            for j in 1:nmode
                norm_val *= abs(ck[j] / bk[j])^idx[j] / factorial(big(idx[j]))^ak
            end
            norm_val = ngam > 0 ? norm_val / ngam : Inf
            
            # Filter condition
            if filter
                filtered = !(norm_val > tolerance && dpt <= ndepth)
            else
                filtered = dpt > ndepth
            end
            
            if !filtered
                push!(idx_queue, copy(idx))
                push!(klm_queue, k)
            end
            
            idx[k] -= 1
        end
        
        if n_idx >= n_est
            break
        end
    end
    
    nado = n_idx
    
    # Build connection matrices
    ado_idx = zeros(Int, nmode, nado)
    idx_plus = fill(-1, nmode, nado)
    idx_minus = fill(-1, nmode, nado)
    
    for n in 1:nado
        idx = nvtmp[:, n]
        ado_idx[:, n] = idx
        
        for k in 1:nmode
            # Forward connection (idx[k] + 1)
            idx[k] += 1
            idx_p = get(idx_hm, idx, -1)
            idx_plus[k, n] = idx_p >= 0 ? idx_p + 1 : -1  # Convert to 1-based
            idx[k] -= 1
            
            # Backward connection (idx[k] - 1)
            idx[k] -= 1
            if idx[k] >= 0
                idx_m = get(idx_hm, idx, -1)
                idx_minus[k, n] = idx_m >= 0 ? idx_m + 1 : -1  # Convert to 1-based
            end
            idx[k] += 1
        end
    end
    
    return nado, ado_idx, idx_plus, idx_minus
end


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
