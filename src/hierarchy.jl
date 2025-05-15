"""
    hierarchy_index_depth(nbranch, ndepth)

Return (nado, nvec, npvec, nmvec) where each branch
index is now 1-based instead of 0-based.
"""
function hierarchy_index_depth(nbranch::Int, ndepth::Int)
    nado = binomial(ndepth + nbranch, nbranch)
    nvec  = zeros(Int, nbranch, nado)
    npvec = zeros(Int, nbranch, nado)
    nmvec = zeros(Int, nbranch, nado)

    jk_temp = zeros(Int, nbranch)
    hierarchy_index_sub!(nbranch, ndepth, jk_temp, 0, 1,
                         nvec, npvec, nmvec)

    return nado, nvec, npvec, nmvec
end


function hierarchy_index_sub!(nbranch::Int, ndepth::Int,
                              jk_temp::Vector{Int},
                              n::Int, k::Int,
                              nvec::Array{Int,2},
                              npvec::Array{Int,2},
                              nmvec::Array{Int,2})
    jk_save = copy(jk_temp)
    for n_prime in n:ndepth
        if k < nbranch
            hierarchy_index_sub!(nbranch, ndepth, jk_temp,
                                 n_prime, k + 1,
                                 nvec, npvec, nmvec)
        else
            index = calc_index(nbranch, jk_temp, n_prime) + 1
            nvec[:, index] = jk_temp

            # 1 step forward
            if n_prime < ndepth
                for k_prime in 1:nbranch
                    jk_p1 = copy(jk_temp)
                    jk_p1[k_prime] += 1
                    npvec[k_prime, index] =
                        calc_index(nbranch, jk_p1, n_prime + 1) + 1
                end
            else
                npvec[:, index] .= -1
            end

            # 1 step backward
            for k_prime in 1:nbranch
                jk_m1 = copy(jk_temp)
                jk_m1[k_prime] -= 1
                nmvec[k_prime, index] =
                    jk_m1[k_prime] ≥ 0 ? 
                    calc_index(nbranch, jk_m1, n_prime - 1) + 1 : -1
            end
        end
        jk_temp[k] += 1
    end

    jk_temp .= jk_save
    return nothing
end


function calc_index(nbranch::Int, jk_target::Vector{Int}, n::Int)
    res = binomial(n + nbranch - 1, nbranch)
    for k in 1:nbranch
        s = sum(jk_target[1:k])
        for i in (s + 1):n
            res += binomial(n - i + nbranch - k - 1, nbranch - k - 1)
        end
    end
    return res
end
