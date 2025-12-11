"""    lsrk4!(P, dt, system)

One step of low-storage RK4 (in-place).
"""
function lsrk4!(P::Matrix{ComplexF64}, dt::Real, system::HEOMSystem;
                liouvillian::Function=liouville!)
    A = (0.0, -1.0, -1.0, -1.0)
    B = (1.0/3.0, 3.0/4.0, 2.0/3.0, 1.0/4.0)
    
    dP = zeros(ComplexF64, size(P))
    tmp = similar(P)
    
    for (α, β) in zip(A, B)
        liouvillian(tmp, P, system)
        @. dP = α * dP + dt * tmp
        @. P += β * dP
    end
    
    return nothing
end

"""    lsrk4(P, dt, system) → P_new

One step of low-storage RK4 (allocating).
"""
function lsrk4(P::Matrix{ComplexF64}, dt::Real, system::HEOMSystem;
               liouvillian::Function=liouville!)
    P_new = copy(P)
    lsrk4!(P_new, dt, system; liouvillian=liouvillian)
    return P_new
end

"""    evolve(system, P0, tspan, dt; savefile=nothing) → (times, populations)

Perform HEOM time evolution.
"""
function evolve(system::HEOMSystem, P0::Matrix{ComplexF64}, 
                tspan::Tuple{Real,Real}, dt::Real;
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
        lsrk4!(P, dt, system; liouvillian=liouvillian)
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


