#=
Compare hierarchy construction methods:
1. Depth-based (hierarchy_index_depth) - full hierarchy up to ndepth
2. Filtered (hierarchy_index_width with filter=true) - norm-based filtering

Both HEOM and HSEOM cases are tested.
=#

using KaisouEOM
using LinearAlgebra
using Printf

println("="^70)
println("Comparison of Hierarchy Construction Methods")
println("="^70)
println()

# =============================================================================
# Part 1: HEOM (Exponential expansion, diagonal decay)
# =============================================================================
println("="^70)
println("Part 1: HEOM (Exponential expansion)")
println("="^70)
println()

# System parameters
ε = 1.0       # Energy gap
Δ = 1.0       # Coupling
H_S = [ε/2 Δ; Δ -ε/2]  # Two-level system Hamiltonian

# Bath parameters (Drude-Lorentz spectral density)
λ = 0.5       # Reorganization energy
γ_c = 1.0     # Cutoff frequency
T = 1.0       # Temperature
β = 1.0 / T   # Inverse temperature

# Matsubara expansion: C(t) = Σₖ cₖ exp(-γₖ t)
# For Drude: c₀ = λγ_c(cot(βγ_c/2) - i), γ₀ = γ_c
#           cₖ = 4λγ_c νₖ / (β(νₖ² - γ_c²)), γₖ = νₖ (Matsubara frequencies)
# where νₖ = 2πk/β

nmatsubara = 3  # Number of Matsubara terms

# Build exponential expansion
γ_list = ComplexF64[]
c_list = ComplexF64[]

# Drude pole
push!(γ_list, γ_c)
push!(c_list, λ * γ_c * (cot(β * γ_c / 2) - 1im))

# Matsubara terms
for k in 1:nmatsubara
    νₖ = 2π * k / β
    push!(γ_list, νₖ)
    push!(c_list, 4λ * γ_c * νₖ / (β * (νₖ^2 - γ_c^2)))
end

# Add complex conjugates for real correlation function
γ_full = vcat(γ_list, conj.(γ_list))
c_full = vcat(c_list, conj.(c_list))

nmode_heom = length(γ_full)
V = ComplexF64.([0 1; 1 0])  # System-bath coupling operator
S_norm_heom = opnorm(V)

println("HEOM Parameters:")
println("  Number of modes: $nmode_heom")
println("  System-bath coupling norm: $S_norm_heom")
println("  Decay rates γ: ", round.(real.(γ_full), digits=3))
println("  |c| coefficients: ", round.(abs.(c_full), digits=3))
println()

# Compare hierarchy construction for different depths
println("-"^50)
println("Hierarchy size comparison (HEOM):")
println("-"^50)
@printf("%-8s  %-12s  %-12s  %-8s\n", "ndepth", "depth-based", "filtered", "ratio")
@printf("%-8s  %-12s  %-12s  %-8s\n", "", "nado", "nado", "")
println("-"^50)

for ndepth in [4, 6, 8, 10, 12]
    # Depth-based (full hierarchy)
    nado_depth, _, _, _ = hierarchy_index_depth(nmode_heom, ndepth)
    
    # Filtered hierarchy
    nado_filter, _, _, _ = hierarchy_index_width(nmode_heom, ndepth;
        γ=real.(γ_full), c=c_full, S_norm=S_norm_heom,
        tolerance=1e-6, filter=true)
    
    ratio = nado_filter / nado_depth
    @printf("%-8d  %-12d  %-12d  %-8.3f\n", ndepth, nado_depth, nado_filter, ratio)
end
println()

# Effect of tolerance
println("-"^50)
println("Effect of tolerance (ndepth=10):")
println("-"^50)
@printf("%-12s  %-12s  %-8s\n", "tolerance", "nado", "ratio")
println("-"^50)

nado_full, _, _, _ = hierarchy_index_depth(nmode_heom, 10)
for tol in [1e-3, 1e-4, 1e-5, 1e-6, 1e-8]
    nado_f, _, _, _ = hierarchy_index_width(nmode_heom, 10;
        γ=real.(γ_full), c=c_full, S_norm=S_norm_heom,
        tolerance=tol, filter=true)
    @printf("%-12.0e  %-12d  %-8.3f\n", tol, nado_f, nado_f/nado_full)
end
println()


# =============================================================================
# Part 2: HSEOM (General basis expansion, non-diagonal D)
# =============================================================================
println()
println("="^70)
println("Part 2: HSEOM (General basis expansion)")
println("="^70)
println()

# PSWF or Chebyshev-like basis with non-diagonal coupling
# Example: Tridiagonal D matrix (common for orthogonal polynomial bases)

function build_tridiagonal_D(nmode::Int, α::Real, β_diag::Real, β_off::Real)
    # D_kk = -α - β_diag * k
    # D_{k,k±1} = β_off * √(k)
    D = zeros(ComplexF64, nmode, nmode)
    for k in 1:nmode
        D[k,k] = -α - β_diag * (k-1)
        if k > 1
            D[k,k-1] = β_off * sqrt(k-1)
        end
        if k < nmode
            D[k,k+1] = β_off * sqrt(k)
        end
    end
    return D
end

# Build HSEOM D matrix
nmode_hseom = 8
α_hseom = 1.0
β_diag_hseom = 0.5
β_off_hseom = 0.3

D_hseom = build_tridiagonal_D(nmode_hseom, α_hseom, β_diag_hseom, β_off_hseom)

# BCF coefficients (example: decaying with mode index)
c_hseom = ComplexF64[1.0 * exp(-0.5 * (k-1)) for k in 1:nmode_hseom]

# System-bath coupling
V_hseom = ComplexF64.([0 1; 1 0])
S_norm_hseom = opnorm(V_hseom)

println("HSEOM Parameters:")
println("  Number of modes: $nmode_hseom")
println("  D matrix structure: tridiagonal")
println("  Diagonal D[k,k]: ", round.(real.(diag(D_hseom)), digits=3))
println("  Off-diagonal |D[k,k+1]|: ", round.([abs(D_hseom[k,k+1]) for k in 1:nmode_hseom-1], digits=3))
println("  |c| coefficients: ", round.(abs.(c_hseom), digits=3))
println()

# Compare hierarchy construction
println("-"^50)
println("Hierarchy size comparison (HSEOM):")
println("-"^50)
@printf("%-8s  %-12s  %-12s  %-8s\n", "ndepth", "depth-based", "filtered", "ratio")
@printf("%-8s  %-12s  %-12s  %-8s\n", "", "nado", "nado", "")
println("-"^50)

for ndepth in [3, 4, 5, 6, 7, 8]
    # Depth-based
    nado_depth, _, _, _ = hierarchy_index_depth(nmode_hseom, ndepth)
    
    # Filtered (with D matrix)
    nado_filter, _, _, _ = hierarchy_index_width(nmode_hseom, ndepth;
        D=D_hseom, c=c_hseom, S_norm=S_norm_hseom,
        tolerance=1e-6, filter=true)
    
    ratio = nado_filter / nado_depth
    @printf("%-8d  %-12d  %-12d  %-8.3f\n", ndepth, nado_depth, nado_filter, ratio)
end
println()

# Effect of tolerance (HSEOM)
println("-"^50)
println("Effect of tolerance (ndepth=6):")
println("-"^50)
@printf("%-12s  %-12s  %-8s\n", "tolerance", "nado", "ratio")
println("-"^50)

nado_full_hseom, _, _, _ = hierarchy_index_depth(nmode_hseom, 6)
for tol in [1e-3, 1e-4, 1e-5, 1e-6, 1e-8]
    nado_f, _, _, _ = hierarchy_index_width(nmode_hseom, 6;
        D=D_hseom, c=c_hseom, S_norm=S_norm_hseom,
        tolerance=tol, filter=true)
    @printf("%-12.0e  %-12d  %-8.3f\n", tol, nado_f, nado_f/nado_full_hseom)
end
println()


# =============================================================================
# Part 3: Using NoiseExp and NoiseGeneral interfaces
# =============================================================================
println()
println("="^70)
println("Part 3: Using NoiseExp/NoiseGeneral interfaces")
println("="^70)
println()

# Create NoiseExp
println("NoiseExp interface:")
bath_exp = BathExp(γ_list, c_list, V)
noise_exp = NoiseExp([bath_exp])

nado_exp_full, _, _, _ = hierarchy_index_width(noise_exp, 8; filter=false)
nado_exp_filt, _, _, _ = hierarchy_index_width(noise_exp, 8; tolerance=1e-5, filter=true)
println("  ndepth=8, filter=false: nado = $nado_exp_full")
println("  ndepth=8, filter=true (tol=1e-5): nado = $nado_exp_filt")
println("  Reduction: $(round(100*(1 - nado_exp_filt/nado_exp_full), digits=1))%")
println()

# Create NoiseGeneral
println("NoiseGeneral interface:")
phi0_hseom = ones(ComplexF64, nmode_hseom)
bath_gen = BathGeneral(D_hseom, phi0_hseom, c_hseom, V_hseom)
noise_gen = NoiseGeneral([bath_gen])

nado_gen_full, _, _, _ = hierarchy_index_width(noise_gen, 5; filter=false)
nado_gen_filt, _, _, _ = hierarchy_index_width(noise_gen, 5; tolerance=1e-5, filter=true)
println("  ndepth=5, filter=false: nado = $nado_gen_full")
println("  ndepth=5, filter=true (tol=1e-5): nado = $nado_gen_filt")
println("  Reduction: $(round(100*(1 - nado_gen_filt/nado_gen_full), digits=1))%")
println()


# =============================================================================
# Part 4: Effective decay rate analysis
# =============================================================================
println()
println("="^70)
println("Part 4: Effective decay rate analysis")
println("="^70)
println()

# Compute effective decay rates for HSEOM
println("HSEOM effective decay rates:")
println("  γₖᵉᶠᶠ = -(Re(Dₖₖ) + Σₗ≠ₖ |Dₖₗ|)")
println()

γ_eff_hseom = zeros(Float64, nmode_hseom)
for k in 1:nmode_hseom
    γ_eff_hseom[k] = -real(D_hseom[k,k])
    for ℓ in 1:nmode_hseom
        if ℓ != k
            γ_eff_hseom[k] -= abs(D_hseom[k,ℓ])
        end
    end
end

# Compute filtering weights
a_hseom = zeros(Float64, nmode_hseom)
for k in 1:nmode_hseom
    if γ_eff_hseom[k] > 0
        a_hseom[k] = sqrt(abs(c_hseom[k])) * S_norm_hseom / γ_eff_hseom[k]
    else
        a_hseom[k] = Inf
    end
end

println("-"^60)
@printf("%-6s  %-10s  %-10s  %-10s  %-10s\n", "mode", "Re(D_kk)", "γ_eff", "|c_k|", "a_k")
println("-"^60)
for k in 1:nmode_hseom
    @printf("%-6d  %-10.3f  %-10.3f  %-10.4f  %-10.4f\n", 
            k, real(D_hseom[k,k]), γ_eff_hseom[k], abs(c_hseom[k]), a_hseom[k])
end
println()

println("Note: Modes with large aₖ require more hierarchy levels.")
println("      Filtering removes ADOs with Wₙ = ∏ₖ aₖⁿₖ/√(nₖ!) < tolerance")
println()


# =============================================================================
# Part 5: Connection matrix validation
# =============================================================================
println()
println("="^70)
println("Part 5: Connection matrix validation")
println("="^70)
println()

function validate_connections(ado_idx, idx_plus, idx_minus, nado, nmode)
    errors = 0
    
    for n in 1:nado
        for k in 1:nmode
            # Check idx_plus
            if idx_plus[k, n] != -1
                n_plus = idx_plus[k, n]
                expected = copy(ado_idx[:, n])
                expected[k] += 1
                if ado_idx[:, n_plus] != expected
                    errors += 1
                end
            end
            
            # Check idx_minus
            if idx_minus[k, n] != -1
                n_minus = idx_minus[k, n]
                expected = copy(ado_idx[:, n])
                expected[k] -= 1
                if ado_idx[:, n_minus] != expected
                    errors += 1
                end
            end
        end
    end
    
    return errors
end

# Validate HEOM filtered hierarchy
nado_h, ado_idx_h, idx_plus_h, idx_minus_h = hierarchy_index_width(nmode_heom, 8;
    γ=real.(γ_full), c=c_full, S_norm=S_norm_heom,
    tolerance=1e-5, filter=true)
errors_heom = validate_connections(ado_idx_h, idx_plus_h, idx_minus_h, nado_h, nmode_heom)
println("HEOM filtered hierarchy (nado=$nado_h): connection errors = $errors_heom")

# Validate HSEOM filtered hierarchy  
nado_s, ado_idx_s, idx_plus_s, idx_minus_s = hierarchy_index_width(nmode_hseom, 5;
    D=D_hseom, c=c_hseom, S_norm=S_norm_hseom,
    tolerance=1e-5, filter=true)
errors_hseom = validate_connections(ado_idx_s, idx_plus_s, idx_minus_s, nado_s, nmode_hseom)
println("HSEOM filtered hierarchy (nado=$nado_s): connection errors = $errors_hseom")

println()
if errors_heom == 0 && errors_hseom == 0
    println("✓ All connection matrices are valid!")
else
    println("✗ Connection matrix errors detected!")
end

println()
println("="^70)
println("Comparison complete!")
println("="^70)
