"""    lsrk4!(P, dt, liouvillian!, system; parallel=false)

One step of low-storage RK4 (in-place).
The Liouville operator has the form `liouvillian!(dP, P, system; parallel)`.

# Arguments
- `parallel::Bool`: If true, use multi-threading in liouvillian.
"""
function lsrk4!(P::Matrix{ComplexF64}, dt::Real, liouvillian!::Function, system;
                parallel::Bool=false)
    A = (0.0, -1.0, -1.0, -1.0)
    B = (1.0/3.0, 3.0/4.0, 2.0/3.0, 1.0/4.0)
    
    dP = zeros(ComplexF64, size(P))
    tmp = similar(P)
    
    for (α, β) in zip(A, B)
        liouvillian!(tmp, P, system; parallel=parallel)
        @. dP = α * dP + dt * tmp
        @. P += β * dP
    end
    
    return nothing
end

"""    lsrk4(P, dt, liouvillian!, system; parallel=false) → P_new

One step of low-storage RK4 (allocating).
"""
function lsrk4(P::Matrix{ComplexF64}, dt::Real, liouvillian!::Function, system;
               parallel::Bool=false)
    P_new = copy(P)
    lsrk4!(P_new, dt, liouvillian!, system; parallel=parallel)
    return P_new
end


# =====================================
# HSEOM time evolution (bra/ket separate)
# =====================================

"""    evolve(system::HSEOMSystem, Pb0, Pk0, tspan, dt; ...) → (times, populations)

HSEOM time evolution (bra/ket simultaneous evolution).

Populations are calculated as p_i = Σₙ Pk[i,n] * conj(Pb[i,n]) (sum over all ADWs).

# Arguments
- `system::HSEOMSystem`: HSEOM system
- `Pb0::Matrix{ComplexF64}`: Initial bra-side ADW (ndim × nadw)
- `Pk0::Matrix{ComplexF64}`: Initial ket-side ADW (ndim × nadw)
- `tspan::Tuple{Real,Real}`: Time range (t_start, t_end)
- `dt::Real`: Time step
- `parallel::Bool`: If true, use multi-threading for ADW loop (default: false)
- `normalized::Bool`: If true, use normalized HSEOM (√n coefficients) (default: false)
"""
function evolve(system::HSEOMSystem, Pb0::Matrix{ComplexF64}, Pk0::Matrix{ComplexF64},
                tspan::Tuple{Real,Real}, dt::Real;
                parallel::Bool=false,
                normalized::Bool=false,
                callback=nothing,
                savefile::Union{String,Nothing}=nothing, save_interval::Int=100)
    t_start, t_end = tspan
    nsteps = Int(ceil((t_end - t_start) / dt))
    ndim = system.ndim
    nadw = system.nadw
    
    times = Vector{Float64}(undef, nsteps + 1)
    populations = Matrix{Float64}(undef, ndim, nsteps + 1)
    
    # Prepare file output
    if savefile !== nothing
        io = open(savefile, "w")
        print(io, "# time")
        for i in 1:ndim
            print(io, "\tpop_$i")
        end
        println(io, "\ttotal\tnorm_ket\tnorm_bra\tmax_abs_ket\tmax_abs_bra")
    end
    
    # Initial state
    Pb = copy(Pb0)
    Pk = copy(Pk0)
    t = t_start
    times[1] = t
    
    # Calculate populations: p_i = Σₙ Pk[i,n] * conj(Pb[i,n])
    for i in 1:ndim
        populations[i, 1] = real(sum(Pk[i, n] * conj(Pb[i, n]) for n in 1:nadw))
    end
    
    if savefile !== nothing
        total_pop = sum(populations[:, 1])
        norm_bra = sqrt(real(sum(abs2.(Pb))))
        norm_ket = sqrt(real(sum(abs2.(Pk))))
        max_abs_ket = maximum(abs.(Pk))
        max_abs_bra = maximum(abs.(Pb))
        print(io, t)
        for i in 1:ndim
            print(io, "\t", populations[i, 1])
        end
        println(io, "\t", total_pop, "\t", norm_ket, "\t", norm_bra, "\t", max_abs_ket, "\t", max_abs_bra)
    end
    
    if callback !== nothing
        callback(t, Pb, Pk)
    end
    
    # Select Liouvillian functions
    ket_liouvillian! = normalized ? liouville_ket_normalized! : liouville_ket!
    bra_liouvillian! = normalized ? liouville_bra_normalized! : liouville_bra!
    
    # Time evolution loop
    for step in 1:nsteps
        lsrk4!(Pb, dt, bra_liouvillian!, system; parallel=parallel)
        lsrk4!(Pk, dt, ket_liouvillian!, system; parallel=parallel)
        t += dt
        times[step + 1] = t
        
        # Calculate populations
        for i in 1:ndim
            populations[i, step + 1] = real(sum(Pk[i, n] * conj(Pb[i, n]) for n in 1:nadw))
        end
        
        # File output at specified intervals
        if savefile !== nothing && (step % save_interval == 0 || step == nsteps)
            total_pop = sum(populations[:, step + 1])
            norm_bra = sqrt(real(sum(abs2.(Pb))))
            norm_ket = sqrt(real(sum(abs2.(Pk))))
            max_abs_ket = maximum(abs.(Pk))
            max_abs_bra = maximum(abs.(Pb))
            print(io, t)
            for i in 1:ndim
                print(io, "\t", populations[i, step + 1])
            end
            println(io, "\t", total_pop, "\t", norm_ket, "\t", norm_bra, "\t", max_abs_ket, "\t", max_abs_bra)
            flush(io)
        end
        
        if callback !== nothing
            callback(t, Pb, Pk)
        end
    end
    
    if savefile !== nothing
        close(io)
    end
    
    return times, populations
end


# =====================================
# HEOM time evolution (density matrix)
# =====================================

"""    evolve(system::HEOMSystem, P0, tspan, dt; ...) → (times, populations)

Perform HEOM time evolution.

# Arguments
- `parallel::Bool`: If true, use multi-threading for ADO loop (default: false)
"""
function evolve(system::HEOMSystem, P0::Matrix{ComplexF64}, 
                tspan::Tuple{Real,Real}, dt::Real;
                parallel::Bool=false,
                liouvillian::Function=liouville!, callback=nothing,
                savefile::Union{String,Nothing}=nothing, save_interval::Int=100)
    t_start, t_end = tspan
    nsteps = Int(ceil((t_end - t_start) / dt))
    ndim = system.ndim
    
    times = Vector{Float64}(undef, nsteps + 1)
    populations = Matrix{Float64}(undef, ndim, nsteps + 1)
    
    # Prepare file output
    if savefile !== nothing
        io = open(savefile, "w")
        print(io, "# time")
        for i in 1:ndim
            print(io, "\tpop_$i")
        end
        println(io)
    end
    
    # Initial state
    P = copy(P0)
    t = t_start
    times[1] = t
    
    # Extract diagonal elements
    for i in 1:ndim
        idx = (i - 1) * ndim + i
        populations[i, 1] = real(P[idx, 1])
    end
    
    if savefile !== nothing
        print(io, t)
        for i in 1:ndim
            print(io, "\t", populations[i, 1])
        end
        println(io)
    end
    
    if callback !== nothing
        callback(t, P)
    end
    
    # Time evolution loop
    for step in 1:nsteps
        lsrk4!(P, dt, liouvillian, system; parallel=parallel)
        t += dt
        times[step + 1] = t
        
        for i in 1:ndim
            idx = (i - 1) * ndim + i
            populations[i, step + 1] = real(P[idx, 1])
        end
        
        # File output at specified intervals
        if savefile !== nothing && (step % save_interval == 0 || step == nsteps)
            print(io, t)
            for i in 1:ndim
                print(io, "\t", populations[i, step + 1])
            end
            println(io)
            flush(io)
        end
        
        if callback !== nothing
            callback(t, P)
        end
    end
    
    if savefile !== nothing
        close(io)
    end
    
    return times, populations
end
