#=
Compare dynamics with different hierarchy construction methods:
1. Depth-based (full hierarchy)
2. Filtered (norm-based filtering)

Tests both HEOM and HSEOM dynamics.
=#

using HEOMKit
using LinearAlgebra
using Printf

println("="^70)
println("Comparison of Dynamics: Depth vs Filtered Hierarchy")
println("="^70)
println()

# =============================================================================
# Part 1: HEOM Dynamics Comparison
# =============================================================================
println("="^70)
println("Part 1: HEOM Dynamics (Spin-Boson Model)")
println("="^70)
println()

# System: Two-level system
ε = 1.0       # Energy gap
Δ = 1.0       # Coupling strength
H_S = ComplexF64[ε/2 Δ; Δ -ε/2]

# Bath: Drude-Lorentz spectral density
λ = 0.5       # Reorganization energy
γ_c = 1.0     # Cutoff frequency
T = 1.0       # Temperature
β = 1.0 / T

# Build Matsubara expansion
nmatsubara = 2

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

V = ComplexF64[0 1; 1 0]

# Create bath and noise
bath = BathExp(γ_list, c_list, V)
noise = NoiseExp([bath])

nmode = noise.nterms
ndim = 2

println("HEOM Parameters:")
println("  ε = $ε, Δ = $Δ")
println("  λ = $λ, γ_c = $γ_c, T = $T")
println("  Number of modes: $nmode")
println()

# Time evolution parameters
tspan = (0.0, 20.0)
dt = 0.01

# Initial state: |↑⟩⟨↑|
ρ0 = ComplexF64[1 0; 0 0]

println("-"^50)
println("Comparing hierarchy sizes and dynamics:")
println("-"^50)

# Different depths and tolerances to compare
test_cases_heom = [
    (ndepth=6, filter=false, tol=0.0, label="depth=6 (full)"),
    (ndepth=8, filter=false, tol=0.0, label="depth=8 (full)"),
    (ndepth=10, filter=false, tol=0.0, label="depth=10 (full)"),
    (ndepth=12, filter=true, tol=1e-4, label="filtered (tol=1e-4)"),
    (ndepth=12, filter=true, tol=1e-5, label="filtered (tol=1e-5)"),
    (ndepth=12, filter=true, tol=1e-6, label="filtered (tol=1e-6)"),
]

# Reference: highest accuracy
ref_ndepth = 12
ref_system = HEOMSystem(H_S, noise, ref_ndepth)
nado_ref = ref_system.nado

# Initial condition for reference
P0_ref = zeros(ComplexF64, ndim^2, nado_ref)
P0_ref[:, 1] = vec(ρ0)

println("Reference: depth=$ref_ndepth (nado=$nado_ref)")
println()

# Run reference
print("Computing reference dynamics... ")
t_ref, pop_ref = evolve(ref_system, P0_ref, tspan, dt)
println("done")
println()

# Store results
results_heom = []

for case in test_cases_heom
    # Build hierarchy
    if case.filter
        nado, ado_idx, idx_plus, idx_minus = hierarchy_index_width(nmode, case.ndepth;
            γ=real.(noise.expon), c=noise.coeff, S_norm=opnorm(V),
            tolerance=case.tol, filter=true)
        
        # Create HEOMOperators using proper constructor
        operators = HEOMOperators(noise, ado_idx, nado)
        
        # Create HEOMMatrices using proper constructor
        heom_mats = HEOMMatrices(H_S, noise; sparse=false)
        
        # Use correct constructor signature
        filtered_system = HEOMSystem(
            noise,
            heom_mats,
            operators,
            ado_idx,
            idx_plus,
            idx_minus,
            nado,
            ndim,
            ndim^2
        )
        
        P0 = zeros(ComplexF64, ndim^2, nado)
        P0[:, 1] = vec(ρ0)
        
        print("$(case.label) (nado=$nado)... ")
        t, pop = evolve(filtered_system, P0, tspan, dt)
        
        # Compute error vs reference
        max_err = maximum(abs.(pop[1, :] .- pop_ref[1, 1:length(pop[1, :])]))
        
        push!(results_heom, (label=case.label, nado=nado, max_err=max_err, times=t, pop=pop))
        println("max error = $(round(max_err, sigdigits=3))")
    else
        system = HEOMSystem(H_S, noise, case.ndepth)
        P0 = zeros(ComplexF64, ndim^2, system.nado)
        P0[:, 1] = vec(ρ0)
        
        print("$(case.label) (nado=$(system.nado))... ")
        t, pop = evolve(system, P0, tspan, dt)
        
        # Compute error vs reference
        max_err = maximum(abs.(pop[1, :] .- pop_ref[1, 1:length(pop[1, :])]))
        
        push!(results_heom, (label=case.label, nado=system.nado, max_err=max_err, times=t, pop=pop))
        println("max error = $(round(max_err, sigdigits=3))")
    end
end

println()
println("-"^50)
println("HEOM Results Summary:")
println("-"^50)
@printf("%-25s  %-8s  %-12s\n", "Method", "nado", "Max Error")
println("-"^50)
for r in results_heom
    @printf("%-25s  %-8d  %-12.2e\n", r.label, r.nado, r.max_err)
end
println()


# =============================================================================
# Part 2: HSEOM Dynamics Comparison
# =============================================================================
println()
println("="^70)
println("Part 2: HSEOM Dynamics (General Basis)")
println("="^70)
println()

# System Hamiltonian (same as HEOM)
H_S_hseom = ComplexF64[ε/2 Δ; Δ -ε/2]

# Build HSEOM with exponential-like expansion (diagonal D matrix for stability)
# This is essentially HEOM but using the HSEOM framework
nterms_hseom = 4
γ_hseom = 1.5  # Characteristic decay rate

# Diagonal D matrix: D_kk = -γ_k (no off-diagonal coupling for stability)
D_hseom = zeros(ComplexF64, nterms_hseom, nterms_hseom)
for k in 1:nterms_hseom
    D_hseom[k, k] = -γ_hseom * k  # Increasing decay rates
end

# BCF coefficients (decaying)
c_hseom = ComplexF64[λ * 0.5^(k-1) for k in 1:nterms_hseom]

# Initial basis function values
phi0_hseom = ones(ComplexF64, nterms_hseom)

V_hseom = ComplexF64[0 1; 1 0]

println("HSEOM Parameters:")
println("  Number of basis functions: $nterms_hseom")
println("  γ base = $γ_hseom")
println("  D matrix: diagonal (exponential-like)")
println("  D diagonals: ", round.(real.(diag(D_hseom)), digits=3))
println()

# Create BathGeneral and NoiseGeneral
bath_gen = BathGeneral(D_hseom, phi0_hseom, c_hseom, V_hseom)
noise_gen = NoiseGeneral([bath_gen])

# Compare different hierarchies
println("-"^50)
println("Comparing HSEOM hierarchy sizes and dynamics:")
println("-"^50)

# Reference with high depth
ref_ndepth_hseom = 8

# Build reference system
nadw_ref, adw_idx_ref, idx_plus_ref, idx_minus_ref = hierarchy_index_depth(nterms_hseom, ref_ndepth_hseom)
hseom_idx_ref = build_hseom_index_maps(adw_idx_ref, nadw_ref, nterms_hseom)

# Create reference NoiseExp-like for HSEOMSystem
noise_exp_ref = NoiseExp(
    zeros(ComplexF64, nterms_hseom),
    c_hseom,
    abs.(c_hseom),
    nterms_hseom,
    1,  # nbath
    [nterms_hseom],
    [1],
    [V_hseom],
    BathExp[]
)

operators_ref = HSEOMOperators(adw_idx_ref)

ref_system_hseom = HSEOMSystem(
    Matrix{ComplexF64}(H_S_hseom),
    [Matrix{ComplexF64}(V_hseom)],
    noise_exp_ref,
    Matrix{ComplexF64}(D_hseom),
    phi0_hseom,
    operators_ref,
    adw_idx_ref,
    idx_plus_ref,
    idx_minus_ref,
    hseom_idx_ref.idx_minus_plus,
    hseom_idx_ref.idx_plus_minus,
    nadw_ref,
    ndim,
    nterms_hseom
)

println("Reference: depth=$ref_ndepth_hseom (nadw=$nadw_ref)")

# Initial state for HSEOM (bra and ket)
Pb0_ref = zeros(ComplexF64, ndim, nadw_ref)
Pk0_ref = zeros(ComplexF64, ndim, nadw_ref)
# |ψ⟩ = |↑⟩, ρ = |↑⟩⟨↑|
Pb0_ref[1, 1] = 1.0  # ⟨↑|
Pk0_ref[1, 1] = 1.0  # |↑⟩

print("Computing reference HSEOM dynamics... ")
t_ref_hseom, pop_ref_hseom = evolve(ref_system_hseom, Pb0_ref, Pk0_ref, tspan, dt)
println("done")
println()

# Test cases for HSEOM
test_cases_hseom = [
    (ndepth=4, filter=false, tol=0.0, label="depth=4 (full)"),
    (ndepth=5, filter=false, tol=0.0, label="depth=5 (full)"),
    (ndepth=6, filter=false, tol=0.0, label="depth=6 (full)"),
    (ndepth=8, filter=true, tol=1e-3, label="filtered (tol=1e-3)"),
    (ndepth=8, filter=true, tol=1e-4, label="filtered (tol=1e-4)"),
    (ndepth=8, filter=true, tol=1e-5, label="filtered (tol=1e-5)"),
]

results_hseom = []

for case in test_cases_hseom
    # Build hierarchy
    if case.filter
        nadw, adw_idx, idx_plus, idx_minus = hierarchy_index_width(nterms_hseom, case.ndepth;
            D=D_hseom, c=c_hseom, S_norm=opnorm(V_hseom),
            tolerance=case.tol, filter=true)
    else
        nadw, adw_idx, idx_plus, idx_minus = hierarchy_index_depth(nterms_hseom, case.ndepth)
    end
    
    # Build two-mode index maps
    hseom_idx = build_hseom_index_maps(adw_idx, nadw, nterms_hseom)
    
    # Create NoiseExp-like for HSEOMSystem
    noise_exp = NoiseExp(
        zeros(ComplexF64, nterms_hseom),
        c_hseom,
        abs.(c_hseom),
        nterms_hseom,
        1,
        [nterms_hseom],
        [1],
        [V_hseom],
        BathExp[]
    )
    
    operators = HSEOMOperators(adw_idx)
    
    system = HSEOMSystem(
        Matrix{ComplexF64}(H_S_hseom),
        [Matrix{ComplexF64}(V_hseom)],
        noise_exp,
        Matrix{ComplexF64}(D_hseom),
        phi0_hseom,
        operators,
        adw_idx,
        idx_plus,
        idx_minus,
        hseom_idx.idx_minus_plus,
        hseom_idx.idx_plus_minus,
        nadw,
        ndim,
        nterms_hseom
    )
    
    # Initial state
    Pb0 = zeros(ComplexF64, ndim, nadw)
    Pk0 = zeros(ComplexF64, ndim, nadw)
    Pb0[1, 1] = 1.0
    Pk0[1, 1] = 1.0
    
    print("$(case.label) (nadw=$nadw)... ")
    t, pop = evolve(system, Pb0, Pk0, tspan, dt)
    
    # Compute error vs reference
    max_err = maximum(abs.(pop[1, :] .- pop_ref_hseom[1, 1:length(pop[1, :])]))
    
    push!(results_hseom, (label=case.label, nadw=nadw, max_err=max_err, times=t, pop=pop))
    println("max error = $(round(max_err, sigdigits=3))")
end

println()
println("-"^50)
println("HSEOM Results Summary:")
println("-"^50)
@printf("%-25s  %-8s  %-12s\n", "Method", "nadw", "Max Error")
println("-"^50)
for r in results_hseom
    @printf("%-25s  %-8d  %-12.2e\n", r.label, r.nadw, r.max_err)
end
println()


# =============================================================================
# Part 3: Output comparison data
# =============================================================================
println()
println("="^70)
println("Part 3: Population Dynamics at Selected Times")
println("="^70)
println()

println("HEOM Population P(↑) at selected times:")
println("-"^60)
@printf("%-8s", "time")
for r in results_heom
    @printf("  %-12s", r.label[1:min(12, length(r.label))])
end
println("  Reference")
println("-"^60)

for t_idx in [1, 201, 501, 1001, 1501, 2001]
    if t_idx <= length(t_ref)
        @printf("%-8.1f", t_ref[t_idx])
        for r in results_heom
            if t_idx <= length(r.times)
                @printf("  %-12.6f", r.pop[1, t_idx])
            else
                @printf("  %-12s", "-")
            end
        end
        @printf("  %-12.6f\n", pop_ref[1, t_idx])
    end
end
println()

println("HSEOM Population P(↑) at selected times:")
println("-"^60)
@printf("%-8s", "time")
for r in results_hseom
    @printf("  %-12s", r.label[1:min(12, length(r.label))])
end
println("  Reference")
println("-"^60)

for t_idx in [1, 201, 501, 1001, 1501, 2001]
    if t_idx <= length(t_ref_hseom)
        @printf("%-8.1f", t_ref_hseom[t_idx])
        for r in results_hseom
            if t_idx <= length(r.times)
                @printf("  %-12.6f", r.pop[1, t_idx])
            else
                @printf("  %-12s", "-")
            end
        end
        @printf("  %-12.6f\n", pop_ref_hseom[1, t_idx])
    end
end
println()


# =============================================================================
# Part 4: Efficiency Analysis
# =============================================================================
println()
println("="^70)
println("Part 4: Efficiency Analysis")
println("="^70)
println()

println("HEOM: Accuracy vs Computational Cost")
println("-"^50)
@printf("%-25s  %-8s  %-12s  %-10s\n", "Method", "nado", "Max Error", "Speedup")
println("-"^50)

nado_baseline = results_heom[end].nado  # filtered with tol=1e-6 as baseline
for r in results_heom
    speedup = r.nado > 0 ? nado_ref / r.nado : 0.0
    @printf("%-25s  %-8d  %-12.2e  %-10.1fx\n", r.label, r.nado, r.max_err, speedup)
end
println()

println("HSEOM: Accuracy vs Computational Cost")
println("-"^50)
@printf("%-25s  %-8s  %-12s  %-10s\n", "Method", "nadw", "Max Error", "Speedup")
println("-"^50)

for r in results_hseom
    speedup = r.nadw > 0 ? nadw_ref / r.nadw : 0.0
    @printf("%-25s  %-8d  %-12.2e  %-10.1fx\n", r.label, r.nadw, r.max_err, speedup)
end
println()

println("="^70)
println("Comparison complete!")
println("="^70)
