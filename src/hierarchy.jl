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
