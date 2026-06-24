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

"""    BathExp(expon, coeff, V)

Construct BathExp from exponential parameters.
"""
function BathExp(expon::Vector, coeff::Vector, V::AbstractMatrix)
    expon_full = ComplexF64.(expon)
    coeff_full = ComplexF64.(coeff)
    nterms = length(expon_full)
    return BathExp(expon_full, coeff_full, nterms, ComplexF64.(V))
end


"""
    NoiseExp

Unified noise parameters from multiple baths.

# Fields
- `γ`: Exponents γₖ
- `c1`: First coefficient c1ₖ (for original term contribution)
- `c2`: Second coefficient c2ₖ (for conjugate term contribution)
- `abs_coeff`: |c1ₖ + c2ₖ|
- `nterms`: Total number of terms
- `nbath`: Number of baths
- `nterms_bath`: Number of terms per bath
- `jstart_bath`: Starting index for each bath
- `V`: Coupling operators (per term)
- `baths`: List of BathExp objects

# Notes on c1 and c2
c1ₖ and c2ₖ are computed by matching modes with conjugate exponents:
- c1ₖ = cₖ (coefficient of mode k)
- c2ₖ = cⱼ* where γⱼ = γₖ* (coefficient from the conjugate mode)

This is computed by sorting modes by (real(γ), imag(γ)) descending for both original
and conjugate data, then matching at the same sorted position (ref/bcf.py algorithm).
"""
struct NoiseExp
    γ::Vector{ComplexF64}
    c1::Vector{ComplexF64}
    c2::Vector{ComplexF64}
    abs_coeff::Vector{Float64}
    nterms::Int
    nbath::Int
    nterms_bath::Vector{Int}
    jstart_bath::Vector{Int}
    V::Vector{Matrix{ComplexF64}}
    baths::Vector{BathExp}
end

"""    NoiseExp(baths::Vector{BathExp})

Construct NoiseExp from multiple BathExp objects.
Expands BCF to include conjugate modes and computes c1, c2 coefficients.
"""
function NoiseExp(baths::Vector{BathExp})
    nbath = length(baths)
    
    # First pass: expand each bath and compute sizes
    expanded_data = [_expand_bcf_with_conjugate(b.expon, b.coeff) for b in baths]
    nterms_bath = [length(data[1]) for data in expanded_data]
    
    jstart_bath = ones(Int, nbath)
    for i in 2:nbath
        jstart_bath[i] = jstart_bath[i-1] + nterms_bath[i-1]
    end
    
    # Unified arrays
    nterms = sum(nterms_bath)
    γ = Vector{ComplexF64}(undef, nterms)
    c1 = Vector{ComplexF64}(undef, nterms)
    c2 = Vector{ComplexF64}(undef, nterms)
    
    for (ibath, (bath_γ, bath_c1, bath_c2)) in enumerate(expanded_data)
        idx_start = jstart_bath[ibath]
        idx_end = idx_start + nterms_bath[ibath] - 1
        γ[idx_start:idx_end] = bath_γ
        c1[idx_start:idx_end] = bath_c1
        c2[idx_start:idx_end] = bath_c2
    end
    
    abs_coeff = abs.(c1 .+ c2)
    
    # Expanded V: one per term
    V = Vector{Matrix{ComplexF64}}(undef, nterms)
    for (ibath, bath) in enumerate(baths)
        idx_start = jstart_bath[ibath]
        idx_end = idx_start + nterms_bath[ibath] - 1
        for k in idx_start:idx_end
            V[k] = bath.V
        end
    end
    
    return NoiseExp(γ, c1, c2, abs_coeff, nterms, nbath, nterms_bath, jstart_bath, V, baths)
end

"""
    _expand_bcf_with_conjugate(expon, coeff) -> (γ, c1, c2)

Expand BCF parameters to include conjugate modes, computing c1 and c2.

For each exponent:
- Real (imag ≈ 0): not doubled, c1 = c, c2 = c*
- Complex with conjugate pair (γa, γb=γa*) with coeffs (ca, cb):
  For γa: c1 = ca + cb*, c2 = ca* + cb
  For γb: c1 = cb + ca*, c2 = cb* + ca
- Complex without conjugate pair: add conj(γ), original gets c1=c, c2=0, conjugate gets c1=0, c2=c*

Output is sorted by |c1+c2|/Re(γ) descending, with conjugate pairs adjacent.
"""
function _expand_bcf_with_conjugate(expon::Vector{ComplexF64}, coeff::Vector{ComplexF64})
    n = length(expon)
    tol = 1e-10
    
    # Classify each exponent
    is_real = [abs(imag(e)) < tol for e in expon]
    has_conj_pair = fill(false, n)  # true if conjugate pair exists in list
    conj_pair_idx = zeros(Int, n)   # index of conjugate pair (if exists)
    
    for i in 1:n
        if is_real[i]
            continue
        end
        # Look for conjugate of expon[i] in the list
        for j in 1:n
            if i != j && abs(expon[i] - conj(expon[j])) < tol
                has_conj_pair[i] = true
                conj_pair_idx[i] = j
                break
            end
        end
    end
    
    # Build temporary arrays for sorting
    # Each entry is either a single mode or a conjugate pair
    # Format: (sort_key, [(γ, c1, c2), ...])
    entries = Tuple{Float64, Vector{Tuple{ComplexF64, ComplexF64, ComplexF64}}}[]
    
    processed = fill(false, n)
    
    for i in 1:n
        processed[i] && continue
        
        if is_real[i]
            # Real: single mode
            c1_i = coeff[i]
            c2_i = conj(coeff[i])
            c_total = c1_i + c2_i
            sort_key = abs(c_total) / real(expon[i])
            push!(entries, (sort_key, [(expon[i], c1_i, c2_i)]))
            processed[i] = true
            
        elseif has_conj_pair[i]
            # Conjugate pair: keep together
            j = conj_pair_idx[i]
            processed[j] && continue  # already processed as part of pair
            
            c_p1, c_p2 = coeff[i], coeff[j]
            c1_i = c_p1 + conj(c_p2)
            c2_i = conj(c_p1) + c_p2
            c1_j = c_p2 + conj(c_p1)
            c2_j = conj(c_p2) + c_p1
            
            # Sort key based on the one with positive imaginary part
            if imag(expon[i]) >= 0
                c_total = c1_i + c2_i
                sort_key = abs(c_total) / real(expon[i])
                push!(entries, (sort_key, [(expon[i], c1_i, c2_i), (expon[j], c1_j, c2_j)]))
            else
                c_total = c1_j + c2_j
                sort_key = abs(c_total) / real(expon[j])
                push!(entries, (sort_key, [(expon[j], c1_j, c2_j), (expon[i], c1_i, c2_i)]))
            end
            processed[i] = true
            processed[j] = true
            
        else
            # No conjugate pair: add conjugate mode
            c1_orig = coeff[i]
            c2_orig = zero(ComplexF64)
            c1_conj = zero(ComplexF64)
            c2_conj = conj(coeff[i])
            
            c_total = c1_orig + c2_orig  # = coeff[i]
            sort_key = abs(c_total) / real(expon[i])
            
            # Put positive imag first
            if imag(expon[i]) >= 0
                push!(entries, (sort_key, [(expon[i], c1_orig, c2_orig), (conj(expon[i]), c1_conj, c2_conj)]))
            else
                push!(entries, (sort_key, [(conj(expon[i]), c1_conj, c2_conj), (expon[i], c1_orig, c2_orig)]))
            end
            processed[i] = true
        end
    end
    
    # Sort by sort_key descending
    sort!(entries, by = x -> -x[1])
    
    # Flatten to output arrays
    n_total = sum(length(e[2]) for e in entries)
    γ = Vector{ComplexF64}(undef, n_total)
    c1 = Vector{ComplexF64}(undef, n_total)
    c2 = Vector{ComplexF64}(undef, n_total)
    
    idx = 0
    for (_, modes) in entries
        for (γ_k, c1_k, c2_k) in modes
            idx += 1
            γ[idx] = γ_k
            c1[idx] = c1_k
            c2[idx] = c2_k
        end
    end
    
    return γ, c1, c2
end

function NoiseExp(bath::BathExp)
    return NoiseExp([bath])
end

"""    compute_heom_params(noise) → (bk, ak, cb)

Compute derived HEOM parameters.
"""
function compute_heom_params(noise::NoiseExp)
    nterms = noise.nterms
    
    bk = [real(noise.γ[j]) + abs(imag(noise.γ[j])) for j in 1:nterms]
    ak = 0.5
    
    coeff = noise.c1 .+ noise.c2
    cb = [abs(coeff[j]) / bk[j] for j in 1:nterms]
    
    # Normalization
    max_cb = maximum(cb)
    if max_cb > 1.0
        bk .*= max_cb
        cb ./= max_cb
    end
    
    return bk, ak, cb
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
