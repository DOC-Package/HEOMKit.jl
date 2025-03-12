module Hierarchy

using SetPara            # should define Jmax, Nmax, tolerance, gamk, c, bk, ak, etc.
# using Noise             # if needed
# using CLSQueue, CLSQueueVec, CLSHashMapVec   # replaced by native Julia types here

# Global arrays (translated from Fortran allocatable module variables)
global nvec::Array{Int,2}  # will be allocated as (Jmax+1)×(Ntotal+2)
global nmvec::Array{Int,2} # (Jmax+1)×(Ntotal+1)
global npvec::Array{Int,2} # (Jmax+1)×(Ntotal+1)
global Ntotal::Int         # total index computed later

# ---------------------------------------------------------------------
# set_index_width() translates the Fortran subroutine of the same name.
# We use a Dict with tuple keys to mimic the Fortran hash_map,
# and simple Vectors as queues.
# ---------------------------------------------------------------------
function set_index_width()
    println("start constructing hierarchy")
    
    # create a hash map: key is a tuple version of the index vector, value is an Int.
    idx_hm = Dict{Tuple{Vararg{Int}}, Int}()
    
    kmax = Jmax + 1
    idx = zeros(Int, kmax)  # Fortran idx(1:kmax)
    
    # Compute an estimate for the number of indices.
    n_est = binomial(Nmax + Jmax + 1, Jmax + 1) - 1
    # nvtmp will store the intermediate indices; its size in Fortran was (kmax,0:n_est).
    # Here we allocate a (kmax)×(n_est+1) array (and map Fortran column j to Julia column j+1)
    nvtmp = zeros(Int, kmax, n_est + 1)
    
    # Use simple vectors to serve as queues.
    idx_que = Vector{Vector{Int}}()   # queue for index vectors
    klm_que = Vector{Int}()             # queue for the corresponding integer values
    
    push!(idx_que, copy(idx))
    push!(klm_que, 1)   # Fortran initializes klm = 1
    
    n_idx = 0
    rdpt = 0   # initialize rdpt
    
    while !isempty(idx_que)
        idx = popfirst!(idx_que)
        klm = popfirst!(klm_que)
        
        # Store the current index in the hash map.
        idx_hm[Tuple(idx)] = n_idx
        nvtmp[:, n_idx + 1] = idx
        n_idx += 1
        
        for k in klm:kmax
            idx[k] += 1
            dpt = sum(idx)
            
            # Compute ngam as a weighted sum (casting to Float64 as needed)
            ngam = 0.0
            for j in 1:kmax
                ngam += float(idx[j]) * float(gamk[j])  # assumes gamk[1] corresponds to Fortran gamk(0)
            end
            
            norm_val = 1.0
            for j in 1:kmax
                norm_val *= (abs(c[j] / bk[j])^idx[j]) / (factorial(idx[j])^ak)
            end
            norm_val /= ngam
            
            # The Fortran code tests a condition then immediately forces filtered = .false.
            filtered = false
            if norm_val > tolerance && dpt <= Nmax
                filtered = false
            else
                filtered = true
            end
            #filtered = false  # override as in the Fortran code
            
            if !filtered
                rdpt = max(rdpt, dpt)
                push!(idx_que, copy(idx))
                push!(klm_que, k)
            end
            idx[k] -= 1
        end
        
        if n_idx == n_est
            break
        end
    end
    
    global Ntotal = n_idx - 1
    
    # Allocate npvec and nmvec as matrices of size (Jmax+1)×(Ntotal+1)
    global npvec = zeros(Int, Jmax + 1, Ntotal + 1)
    global nmvec = zeros(Int, Jmax + 1, Ntotal + 1)
    
    # Loop over each stored index (note: Fortran loop n=0:Ntotal is mapped to Julia n+1)
    for n in 0:Ntotal
        idx = copy(nvtmp[:, n + 1])
        for k in 1:kmax
            idx[k] += 1
            # get_or_default: here we use get(dict, key, default)
            npvec[k, n + 1] = get(idx_hm, Tuple(idx), -1)
            idx[k] -= 2
            nmvec[k, n + 1] = get(idx_hm, Tuple(idx), -1)
            idx[k] += 1
        end
    end
    
    # Allocate nvec as a (Jmax+1)×(Ntotal+2) matrix.
    global nvec = zeros(Int, Jmax + 1, Ntotal + 2)
    # In Fortran: nvec(0:Jmax,0:Ntotal) = nvtmp(1:kmax,0:Ntotal)
    # Mapping indices: Fortran’s column 0 becomes Julia’s column 1, etc.
    nvec[:, 1:Ntotal + 1] = nvtmp[:, 1:Ntotal + 1]
end

# ---------------------------------------------------------------------
# set_index_depth() translates the corresponding Fortran subroutine.
# It allocates working arrays and then calls the recursive routine.
# ---------------------------------------------------------------------
function set_index_depth()
    # Reallocate nvec, npvec, nmvec to the sizes expected.
    total = binomial(Nmax + Jmax + 1, Jmax + 1) - 1
    global nvec = zeros(Int, Jmax + 1, total + 2)
    global npvec = zeros(Int, Jmax + 1, total + 1)
    global nmvec = zeros(Int, Jmax + 1, total + 1)
    
    jk_temp = zeros(Int, Jmax + 1)
    create_hierarchy_sub!(jk_temp, 0, 0)
end

# ---------------------------------------------------------------------
# create_hierarchy_sub!() is the recursive subroutine translated from Fortran.
# (The “!” in the name indicates that it modifies its argument.)
# Note that the Fortran code uses 0-based indexing for jk_temp.
# Here we assume that jk_temp[1] corresponds to Fortran jk_temp(0).
# The parameters n and k are passed as integers.
# ---------------------------------------------------------------------
function create_hierarchy_sub!(jk_temp::Vector{Int}, n::Int, k::Int)
    jk_save = copy(jk_temp)
    for n_prime in n:Nmax
        if k < Jmax
            create_hierarchy_sub!(jk_temp, n_prime, k + 1)
        else
            index = calc_index(Jmax, jk_temp, n_prime)
            # Store jk_temp into column "index" of nvec.
            # Since calc_index returns a Fortran-style index (starting at 0),
            # we access column index+1 in Julia.
            nvec[:, index + 1] = jk_temp
            if n_prime < Nmax
                for k_prime in 0:Jmax
                    jk_p1 = copy(jk_temp)
                    jk_p1[k_prime + 1] += 1
                    npvec[k_prime + 1, index + 1] = calc_index(Jmax, jk_p1, n_prime + 1)
                end
            else
                npvec[:, index + 1] .= -1
            end
            
            for k_prime in 0:Jmax
                jk_m1 = copy(jk_temp)
                jk_m1[k_prime + 1] -= 1
                if jk_m1[k_prime + 1] >= 0
                    nmvec[k_prime + 1, index + 1] = calc_index(Jmax, jk_m1, n_prime - 1)
                else
                    nmvec[k_prime + 1, index + 1] = -1
                end
            end
        end
        jk_temp[k + 1] += 1
    end
    jk_temp .= jk_save
end

# ---------------------------------------------------------------------
# calc_index(Jmax, jk_target, n) is the translation of the Fortran function.
# Note the adjustment of indices: Fortran’s jk_target(0:Jmax) becomes
# jk_target[1:Jmax+1] in Julia.
# ---------------------------------------------------------------------
function calc_index(Jmax::Int, jk_target::Vector{Int}, n::Int)
    # Compute the base value using a binomial coefficient.
    res = binomial(n - 1 + Jmax + 1, Jmax + 1)
    for k in 0:Jmax
        s = sum(jk_target[1:k+1])  # sum of jk_target(0:k) in Fortran
        for i in (s + 1):n
            res += binomial(n - i + Jmax - k - 1, Jmax - k - 1)
        end
    end
    return res
end

end  # module Hierarchy
