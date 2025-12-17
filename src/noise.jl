"""
    Bath

Single bath with exponential expansion of correlation function C(t) = Σₖ cₖ exp(-γₖt).

Fields: `expon` (γₖ), `coeff` (cₖ), `nterms`, `V` (coupling operator)
"""
struct Bath
    expon::Vector{ComplexF64}
    coeff::Vector{ComplexF64}
    nterms::Int
    V::Matrix{ComplexF64}
end

"""    Bath(expon, coeff, V; add_conjugate=true)

Construct Bath from exponential parameters. Conjugate pairs added by default.
"""
function Bath(expon::Vector, coeff::Vector, V::AbstractMatrix;
              add_conjugate::Bool=true)
    if add_conjugate
        expon_full = vcat(ComplexF64.(expon), conj.(ComplexF64.(expon)))
        coeff_full = vcat(ComplexF64.(coeff), conj.(ComplexF64.(coeff)))
    else
        expon_full = ComplexF64.(expon)
        coeff_full = ComplexF64.(coeff)
    end
    nterms = length(expon_full)
    return Bath(expon_full, coeff_full, nterms, ComplexF64.(V))
end

"""
    Noise

Unified noise parameters from multiple baths.

Fields: `expon`, `coeff`, `abs_coeff`, `nterms`, `nbath`, `nterms_bath`, `jstart_bath`, `V`, `baths`
"""
struct Noise
    expon::Vector{ComplexF64}
    coeff::Vector{ComplexF64}
    abs_coeff::Vector{Float64}
    nterms::Int
    nbath::Int
    nterms_bath::Vector{Int}
    jstart_bath::Vector{Int}
    V::Vector{Matrix{ComplexF64}}
    baths::Vector{Bath}
end

"""    Noise(baths::Vector{Bath})

Construct Noise from multiple Bath objects.
"""
function Noise(baths::Vector{Bath})
    nbath = length(baths)
    
    # Parameter sizes and start positions for each bath
    nterms_bath = [b.nterms for b in baths]
    jstart_bath = ones(Int, nbath)
    for i in 2:nbath
        jstart_bath[i] = jstart_bath[i-1] + nterms_bath[i-1]
    end
    
    # Unified arrays
    nterms = sum(nterms_bath)
    expon = Vector{ComplexF64}(undef, nterms)
    coeff = Vector{ComplexF64}(undef, nterms)
    
    for (ibath, bath) in enumerate(baths)
        idx_start = jstart_bath[ibath]
        idx_end = idx_start + nterms_bath[ibath] - 1
        expon[idx_start:idx_end] = bath.expon
        coeff[idx_start:idx_end] = bath.coeff
    end
    
    abs_coeff = abs.(coeff)
    V = [b.V for b in baths]
    
    return Noise(expon, coeff, abs_coeff, nterms, nbath, nterms_bath, jstart_bath, V, baths)
end

function Noise(bath::Bath)
    return Noise([bath])
end

"""    compute_heom_params(noise) → (bk, ak, cb)

Compute derived HEOM parameters.
"""
function compute_heom_params(noise::Noise)
    nterms = noise.nterms
    
    bk = [real(noise.expon[j]) + abs(imag(noise.expon[j])) for j in 1:nterms]
    ak = 0.5
    
    cb = [abs(noise.coeff[j]) / bk[j] for j in 1:nterms]
    
    # Normalization
    max_cb = maximum(cb)
    if max_cb > 1.0
        bk .*= max_cb
        cb ./= max_cb
    end
    
    return bk, ak, cb
end


# =====================================
# Display functions
# =====================================

function Base.show(io::IO, bath::Bath)
    print(io, "Bath(nterms=$(bath.nterms))")
end

function Base.show(io::IO, ::MIME"text/plain", bath::Bath)
    println(io, "Bath:")
    println(io, "  nterms = $(bath.nterms)")
    println(io, "  V size = $(size(bath.V))")
    println(io, "  γₖ range: $(minimum(abs.(bath.expon))) - $(maximum(abs.(bath.expon)))")
    println(io, "  cₖ range: $(minimum(abs.(bath.coeff))) - $(maximum(abs.(bath.coeff)))")
end

function Base.show(io::IO, noise::Noise)
    print(io, "Noise(nterms=$(noise.nterms), nbath=$(noise.nbath))")
end

function Base.show(io::IO, ::MIME"text/plain", noise::Noise)
    println(io, "Noise:")
    println(io, "  Total nterms = $(noise.nterms)")
    println(io, "  Number of baths = $(noise.nbath)")
    for i in 1:noise.nbath
        println(io, "    Bath $i: nterms = $(noise.nterms_bath[i]), start = $(noise.jstart_bath[i])")
    end
end