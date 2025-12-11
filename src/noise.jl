"""
    Noise module for HEOM
"""

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

# Note: Bath(sd::SpectralDensity, ...) constructor is available when QFiND is loaded
# See the package extension or use the direct constructor with gamk/ck vectors.

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


# =====================================
# Noise structure (unified multiple baths)
# =====================================

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
    
    # 各熱浴のパラメータサイズと開始位置
    nterms_bath = [b.nterms for b in baths]
    jstart_bath = ones(Int, nbath)
    for i in 2:nbath
        jstart_bath[i] = jstart_bath[i-1] + nterms_bath[i-1]
    end
    
    # 統合配列
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

"""    Noise(bath::Bath)

Construct Noise from a single Bath.
"""
function Noise(bath::Bath)
    return Noise([bath])
end


# =====================================
# Convenience constructors
# =====================================

"""    drude_bath(λ, γ, T, V; degree=10)

Create bath with Drude-Lorentz spectral density J(ω) = 2λγω/(ω² + γ²).
"""
function drude_bath(λ::Real, γ::Real, temperature::Real, V::AbstractMatrix;
                    degree::Int=10, integrator::Symbol=:pade)
    sd = DrudeLorentz(λ=λ, γ=γ)
    return Bath(sd, temperature, V; degree=degree, integrator=integrator)
end

"""    brownian_bath(λ, γ, Ω, T, V; degree=10)

Create bath with Brownian spectral density J(ω) = 4λγΩ²ω/((ω² - Ω²)² + 4γ²ω²).
"""
function brownian_bath(λ::Real, γ::Real, Ω::Real, temperature::Real, V::AbstractMatrix;
                       degree::Int=10, integrator::Symbol=:pade)
    sd = Brownian(λ=λ, γ=γ, Ω=Ω)
    return Bath(sd, temperature, V; degree=degree, integrator=integrator)
end


# =====================================
# Derived parameter computation (for HEOM)
# =====================================

"""    compute_heom_params(noise) → (bk, ak, cb)

Compute derived HEOM parameters.
"""
function compute_heom_params(noise::Noise)
    nterms = noise.nterms
    
    bk = [real(noise.expon[j]) + abs(imag(noise.expon[j])) for j in 1:nterms]
    ak = 0.5
    
    cb = [abs(noise.coeff[j]) / bk[j] for j in 1:nterms]
    
    # 正規化
    max_cb = maximum(cb)
    if max_cb > 1.0
        bk .*= max_cb
        cb ./= max_cb
    end
    
    return bk, ak, cb
end


# =====================================
# 表示用
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