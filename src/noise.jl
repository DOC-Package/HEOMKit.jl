"""
    BathExp

Single bath with exponential expansion of correlation function C(t) = Σₖ cₖ exp(-γₖt).

Fields: `expon` (γₖ), `coeff` (cₖ), `nterms`, `V` (coupling operator)
"""
struct BathExp
    expon::Vector{ComplexF64}
    coeff::Vector{ComplexF64}
    nterms::Int
    V::Matrix{ComplexF64}
end

"""    BathExp(expon, coeff, V; add_conjugate=true)

Construct BathExp from exponential parameters. Conjugate pairs added by default.
"""
function BathExp(expon::Vector, coeff::Vector, V::AbstractMatrix;
              add_conjugate::Bool=true)
    if add_conjugate
        expon_full = vcat(ComplexF64.(expon), conj.(ComplexF64.(expon)))
        coeff_full = vcat(ComplexF64.(coeff), conj.(ComplexF64.(coeff)))
    else
        expon_full = ComplexF64.(expon)
        coeff_full = ComplexF64.(coeff)
    end
    nterms = length(expon_full)
    return BathExp(expon_full, coeff_full, nterms, ComplexF64.(V))
end

"""
    BathGeneral

Single bath with general basis expansion of correlation function.
C(t) = Σₖ cₖ φₖ(t), where ∂ₜφₖ = Σₗ Dₖₗ φₗ

# Fields
- `D::Matrix{ComplexF64}`: Derivative matrix (∂ₜφₖ = Σₗ Dₖₗ φₗ)
- `phi0::Vector{ComplexF64}`: Initial values φₖ(0)
- `coeff::Vector{ComplexF64}`: Expansion coefficients cₖ
- `nterms::Int`: Number of expansion terms
- `V::Matrix{ComplexF64}`: Interaction operator
"""
struct BathGeneral
    D::Matrix{ComplexF64}
    phi0::Vector{ComplexF64}
    coeff::Vector{ComplexF64}
    nterms::Int
    V::Matrix{ComplexF64}
end

"""    BathGeneral(D, phi0, coeff, V)

Construct BathGeneral from derivative matrix and parameters.

# Arguments
- `D::AbstractMatrix`: Derivative matrix (nterms × nterms)
- `phi0::AbstractVector`: Initial values φₖ(0)
- `coeff::AbstractVector`: Expansion coefficients cₖ
- `V::AbstractMatrix`: Interaction operator
"""
function BathGeneral(D::AbstractMatrix, phi0::AbstractVector, coeff::AbstractVector, V::AbstractMatrix)
    nterms = size(D, 1)
    @assert size(D, 2) == nterms "D matrix must be square"
    @assert length(phi0) == nterms "phi0 length must match D matrix size"
    @assert length(coeff) == nterms "coeff length must match D matrix size"
    
    return BathGeneral(
        ComplexF64.(D),
        ComplexF64.(phi0),
        ComplexF64.(coeff),
        nterms,
        ComplexF64.(V)
    )
end

"""
    NoiseExp

Unified noise parameters from multiple baths.

Fields: `expon`, `coeff`, `abs_coeff`, `nterms`, `nbath`, `nterms_bath`, `jstart_bath`, `V`, `baths`
"""
struct NoiseExp
    expon::Vector{ComplexF64}
    coeff::Vector{ComplexF64}
    abs_coeff::Vector{Float64}
    nterms::Int
    nbath::Int
    nterms_bath::Vector{Int}
    jstart_bath::Vector{Int}
    V::Vector{Matrix{ComplexF64}}     # List of interaction operators (nbath)
    baths::Vector{BathExp}                # List of baths
end

"""
    NoiseGeneral

Unified noise parameters from multiple BathGeneral objects.
General basis expansion for HSEOM: C(t) = Σₖ cₖ φₖ(t), where ∂ₜφₖ = Σₗ Dₖₗ φₗ

# Fields
- `D::Matrix{ComplexF64}`: Block-diagonal derivative matrix (unified)
- `phi0::Vector{ComplexF64}`: Initial values φₖ(0) (unified)
- `coeff::Vector{ComplexF64}`: Expansion coefficients cₖ (unified)
- `nterms::Int`: Total number of expansion terms
- `nbath::Int`: Number of baths
- `nterms_bath::Vector{Int}`: Number of terms per bath
- `jstart_bath::Vector{Int}`: Starting index for each bath
- `V::Vector{Matrix{ComplexF64}}`: Interaction operators (per bath)
- `baths::Vector{BathGeneral}`: List of baths
"""
struct NoiseGeneral
    D::Matrix{ComplexF64}
    phi0::Vector{ComplexF64}
    coeff::Vector{ComplexF64}
    nterms::Int
    nbath::Int
    nterms_bath::Vector{Int}
    jstart_bath::Vector{Int}
    V::Vector{Matrix{ComplexF64}}
    baths::Vector{BathGeneral}
end

"""    NoiseGeneral(baths::Vector{BathGeneral})

Construct NoiseGeneral from multiple BathGeneral objects.
"""
function NoiseGeneral(baths::Vector{BathGeneral})
    nbath = length(baths)
    
    # Parameter sizes and start positions for each bath
    nterms_bath = [b.nterms for b in baths]
    jstart_bath = ones(Int, nbath)
    for i in 2:nbath
        jstart_bath[i] = jstart_bath[i-1] + nterms_bath[i-1]
    end
    
    # Total number of terms
    nterms = sum(nterms_bath)
    
    # Build unified arrays
    phi0 = Vector{ComplexF64}(undef, nterms)
    coeff = Vector{ComplexF64}(undef, nterms)
    D = zeros(ComplexF64, nterms, nterms)
    
    for (ibath, bath) in enumerate(baths)
        idx_start = jstart_bath[ibath]
        idx_end = idx_start + nterms_bath[ibath] - 1
        phi0[idx_start:idx_end] = bath.phi0
        coeff[idx_start:idx_end] = bath.coeff
        D[idx_start:idx_end, idx_start:idx_end] = bath.D
    end
    
    V = [b.V for b in baths]
    
    return NoiseGeneral(D, phi0, coeff, nterms, nbath, nterms_bath, jstart_bath, V, baths)
end

function NoiseGeneral(bath::BathGeneral)
    return NoiseGeneral([bath])
end

"""    NoiseExp(baths::Vector{BathExp})

Construct NoiseExp from multiple BathExp objects.
"""
function NoiseExp(baths::Vector{BathExp})
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
    
    return NoiseExp(expon, coeff, abs_coeff, nterms, nbath, nterms_bath, jstart_bath, V, baths)
end

function NoiseExp(bath::BathExp)
    return NoiseExp([bath])
end

"""    compute_heom_params(noise) → (bk, ak, cb)

Compute derived HEOM parameters.
"""
function compute_heom_params(noise::NoiseExp)
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

function Base.show(io::IO, bath::BathExp)
    print(io, "BathExp(nterms=$(bath.nterms))")
end

function Base.show(io::IO, ::MIME"text/plain", bath::BathExp)
    println(io, "BathExp:")
    println(io, "  nterms = $(bath.nterms)")
    println(io, "  V size = $(size(bath.V))")
    println(io, "  γₖ range: $(minimum(abs.(bath.expon))) - $(maximum(abs.(bath.expon)))")
    println(io, "  cₖ range: $(minimum(abs.(bath.coeff))) - $(maximum(abs.(bath.coeff)))")
end

function Base.show(io::IO, bath::BathGeneral)
    print(io, "BathGeneral(nterms=$(bath.nterms))")
end

function Base.show(io::IO, ::MIME"text/plain", bath::BathGeneral)
    println(io, "BathGeneral:")
    println(io, "  nterms = $(bath.nterms)")
    println(io, "  V size = $(size(bath.V))")
    println(io, "  D matrix size = $(size(bath.D))")
    println(io, "  φₖ(0) range: $(minimum(abs.(bath.phi0))) - $(maximum(abs.(bath.phi0)))")
    println(io, "  cₖ range: $(minimum(abs.(bath.coeff))) - $(maximum(abs.(bath.coeff)))")
end

function Base.show(io::IO, noise::NoiseExp)
    print(io, "NoiseExp(nterms=$(noise.nterms), nbath=$(noise.nbath))")
end

function Base.show(io::IO, ::MIME"text/plain", noise::NoiseExp)
    println(io, "NoiseExp:")
    println(io, "  Total nterms = $(noise.nterms)")
    println(io, "  Number of baths = $(noise.nbath)")
    for i in 1:noise.nbath
        println(io, "    Bath $i: nterms = $(noise.nterms_bath[i]), start = $(noise.jstart_bath[i])")
    end
end

function Base.show(io::IO, noise::NoiseGeneral)
    print(io, "NoiseGeneral(nterms=$(noise.nterms), nbath=$(noise.nbath))")
end

function Base.show(io::IO, ::MIME"text/plain", noise::NoiseGeneral)
    println(io, "NoiseGeneral:")
    println(io, "  Total nterms = $(noise.nterms)")
    println(io, "  Number of baths = $(noise.nbath)")
    println(io, "  D matrix size = $(size(noise.D))")
    println(io, "  φₖ(0) range: $(minimum(abs.(noise.phi0))) - $(maximum(abs.(noise.phi0)))")
    println(io, "  cₖ range: $(minimum(abs.(noise.coeff))) - $(maximum(abs.(noise.coeff)))")
    for i in 1:noise.nbath
        println(io, "    Bath $i: nterms = $(noise.nterms_bath[i]), start = $(noise.jstart_bath[i])")
    end
end