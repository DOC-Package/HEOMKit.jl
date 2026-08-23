#=
Compare hierarchy construction methods for HSEOM with PSWF expansion
Based on examples/hseom/semicircle/semicircle.jl

This compares:
1. Depth-based hierarchy (hierarchy_index_depth)
2. Filtered hierarchy (hierarchy_index_width with filter=true)

Using the same physical parameters as the semicircle example.
=#

using HEOMKit
using HEOMKit: icm2ifs, kB
using QFiND
using ProlateSpheroidalWaveFunctions
using LinearAlgebra
using Printf

println("="^70)
println("HSEOM Hierarchy Construction Comparison (PSWF Expansion)")
println("="^70)

# =============================================
# 1. Setup: Same parameters as semicircle.jl
# =============================================
println("\n" * "="^60)
println("1. Setting up spectral density and PSWF expansion")
println("="^60)

# Spectral density parameters
s = 1.0          # Ohmic
γc = 50.0        # Cutoff frequency [cm⁻¹]
λ = 5.0          # Reorganization energy [cm⁻¹]
T = 300.0        # Temperature [K]

sd = SemicircleSD(s, γc, λ)

println("\nSpectral Density Parameters:")
println("  Type: Ohmic (s=$s)")
println("  Cutoff frequency γc = $γc cm⁻¹")
println("  Reorganization energy λ = $λ cm⁻¹")
println("  Temperature T = $T K")

# PSWF expansion parameters
ω_min = -50.0
ω_max = 50.0
n_terms = 8
T_pswf = 200.0

# System Hamiltonian
ε = 0.0
Δ = 20.0
H = ComplexF64[ε/2 Δ; Δ -ε/2] * icm2ifs
ndim = 2

println("\nPSWF Expansion Parameters:")
println("  Number of terms: $n_terms")
println("  Time duration: $T_pswf fs")

# Construct PSWF expansion
sbeta = BosonicQNSD(sd, T)
f = w -> sbeta(w; scale=icm2ifs) / pi
pswfft = pswf_expansion_fourier(f, ω_min * icm2ifs, ω_max * icm2ifs, T_pswf, n_terms; pm=-1.0)

# Extract c and D
c_coeffs = pswfft.coeffs
D_matrix = compute_time_derivative_matrix(pswfft)

println("\nExpansion coefficients |c| (first 5):")
for k in 1:min(5, n_terms)
    println("  |c[$k]| = $(round(abs(c_coeffs[k]), sigdigits=4))")
end

# Interaction operator
V = ComplexF64[1 0; 0 -1]
S_norm = opnorm(V)

# φₖ(0) for PSWF expansion
phi0 = zeros(ComplexF64, n_terms)
for k in 1:n_terms
    phi0[k] = pswfft.basis[k](0.0)
end

# Create noise
bath = BathGeneral(D_matrix, phi0, c_coeffs, V)
noise = NoiseGeneral(bath)

# =============================================
# 2. Analyze filtering parameters
# =============================================
println("\n" * "="^60)
println("2. Filtering parameter analysis")
println("="^60)

# For HSEOM (PSWF), filtering uses aₖ = √|cₖ| × ‖S‖_max (no γ division)
# because D_kk is purely imaginary (oscillatory, not decaying)
a = sqrt.(abs.(c_coeffs)) .* S_norm

println("\nMode analysis (HSEOM uses aₖ = √|cₖ| × ‖S‖_max):")
println("-"^60)
@printf("%-6s  %-12s  %-12s  %-12s\n", "mode", "Re(D_kk)", "|c_k|", "a_k")
println("-"^60)
for k in 1:n_terms
    @printf("%-6d  %-12.4f  %-12.4e  %-12.4e\n", 
            k, real(D_matrix[k,k]), abs(c_coeffs[k]), a[k])
end
println()

println("Note: D_kk is purely imaginary for PSWF (oscillatory basis)")
println("      → Filtering is based on |c_k| only, not decay rate")

# =============================================
# 3. Compare hierarchy sizes
# =============================================
println("\n" * "="^60)
println("3. Hierarchy size comparison")
println("="^60)

# Use appropriate tolerances for PSWF (coefficients are small ~1e-5)
tolerances = [1e-6, 1e-8, 1e-10, 1e-12]

println("\nHierarchy sizes for different depths and tolerances:")
println("-"^70)
@printf("%-8s  %-10s  ", "ndepth", "full")
for tol in tolerances
    @printf("%-12s  ", "tol=$(tol)")
end
println()
println("-"^70)

for ndepth in [4, 5, 6, 7, 8]
    # Full hierarchy
    nado_full, _, _, _ = hierarchy_index_depth(n_terms, ndepth)
    @printf("%-8d  %-10d  ", ndepth, nado_full)
    
    # Filtered hierarchies
    for tol in tolerances
        nado_filt, _, _, _ = hierarchy_index_width(n_terms, ndepth;
            D=D_matrix, c=c_coeffs, S_norm=S_norm,
            tolerance=tol, filter=true)
        ratio = nado_filt / nado_full
        @printf("%-4d(%4.1f%%)  ", nado_filt, ratio*100)
    end
    println()
end
println()

# =============================================
# 4. Dynamics comparison
# =============================================
println("\n" * "="^60)
println("4. Dynamics comparison")
println("="^60)

# Time evolution parameters
t_end = 200.0
dt = 0.25
tspan = (0.0, t_end)

println("\nTime Evolution Parameters:")
println("  Time range: [0, $t_end] fs")
println("  Time step: $dt fs")

# Helper function to create HSEOM system with custom hierarchy
function create_hseom_system(H, noise, D_matrix, phi0, V, ado_idx, idx_plus, idx_minus, nado, ndim, nterms)
    hseom_idx = build_hseom_index_maps(ado_idx, nado, nterms)
    
    noise_exp = NoiseExp(
        zeros(ComplexF64, nterms),
        noise.coeff,
        abs.(noise.coeff),
        nterms,
        1,
        [nterms],
        [1],
        [V],
        BathExp[]
    )
    
    operators = HSEOMOperators(ado_idx)
    
    return HSEOMSystem(
        Matrix{ComplexF64}(H),
        [Matrix{ComplexF64}(V)],
        noise_exp,
        Matrix{ComplexF64}(D_matrix),
        phi0,
        operators,
        ado_idx,
        idx_plus,
        idx_minus,
        hseom_idx.idx_minus_plus,
        hseom_idx.idx_plus_minus,
        nado,
        ndim,
        nterms
    )
end

# Reference: high depth
ref_ndepth = 8
nado_ref, ado_idx_ref, idx_plus_ref, idx_minus_ref = hierarchy_index_depth(n_terms, ref_ndepth)
system_ref = create_hseom_system(H, noise, D_matrix, phi0, V, 
                                  ado_idx_ref, idx_plus_ref, idx_minus_ref, 
                                  nado_ref, ndim, n_terms)

println("\nReference: depth=$ref_ndepth (nadw=$nado_ref)")

# Initial state
Pb0_ref = zeros(ComplexF64, ndim, nado_ref)
Pk0_ref = zeros(ComplexF64, ndim, nado_ref)
Pb0_ref[1, 1] = 1.0
Pk0_ref[1, 1] = 1.0

print("Computing reference dynamics... ")
t_ref, pop_ref = evolve(system_ref, Pb0_ref, Pk0_ref, tspan, dt; normalized=true)
println("done")
println()

# Test cases - use appropriate tolerances for PSWF
test_cases = [
    (ndepth=5, filter=false, tol=0.0, label="depth=5 (full)"),
    (ndepth=6, filter=false, tol=0.0, label="depth=6 (full)"),
    (ndepth=7, filter=false, tol=0.0, label="depth=7 (full)"),
    (ndepth=8, filter=true, tol=1e-8, label="filtered (tol=1e-8)"),
    (ndepth=8, filter=true, tol=1e-10, label="filtered (tol=1e-10)"),
    (ndepth=8, filter=true, tol=1e-12, label="filtered (tol=1e-12)"),
]

results = []

for case in test_cases
    # Build hierarchy
    if case.filter
        nado, ado_idx, idx_plus, idx_minus = hierarchy_index_width(n_terms, case.ndepth;
            D=D_matrix, c=c_coeffs, S_norm=S_norm,
            tolerance=case.tol, filter=true)
    else
        nado, ado_idx, idx_plus, idx_minus = hierarchy_index_depth(n_terms, case.ndepth)
    end
    
    # Create system
    system = create_hseom_system(H, noise, D_matrix, phi0, V,
                                  ado_idx, idx_plus, idx_minus,
                                  nado, ndim, n_terms)
    
    # Initial state
    Pb0 = zeros(ComplexF64, ndim, nado)
    Pk0 = zeros(ComplexF64, ndim, nado)
    Pb0[1, 1] = 1.0
    Pk0[1, 1] = 1.0
    
    print("$(case.label) (nadw=$nado)... ")
    t, pop = evolve(system, Pb0, Pk0, tspan, dt; normalized=true)
    
    # Compute error
    max_err = maximum(abs.(pop[1, :] .- pop_ref[1, 1:length(pop[1, :])]))
    
    push!(results, (label=case.label, nadw=nado, max_err=max_err, times=t, pop=pop))
    println("max error = $(round(max_err, sigdigits=3))")
end

# =============================================
# 5. Results Summary
# =============================================
println("\n" * "="^60)
println("5. Results Summary")
println("="^60)

println("\nAccuracy vs Computational Cost:")
println("-"^60)
@printf("%-25s  %-8s  %-12s  %-10s\n", "Method", "nadw", "Max Error", "Speedup")
println("-"^60)
for r in results
    speedup = nado_ref / r.nadw
    @printf("%-25s  %-8d  %-12.2e  %-10.1fx\n", r.label, r.nadw, r.max_err, speedup)
end
println()

# Population at selected times
println("\nPopulation P(1) at selected times:")
println("-"^80)
@printf("%-8s", "time")
for r in results
    @printf("  %-12s", r.label[1:min(12, length(r.label))])
end
println("  Reference")
println("-"^80)

for t_idx in [1, 101, 201, 401, 601, 801]
    if t_idx <= length(t_ref)
        @printf("%-8.1f", t_ref[t_idx])
        for r in results
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

# =============================================
# 6. Key findings
# =============================================
println("\n" * "="^60)
println("6. Key Findings")
println("="^60)

# Find best filtered result
filtered_results = filter(r -> occursin("filtered", r.label), results)
if !isempty(filtered_results)
    idx = argmin([r.max_err for r in filtered_results])
    best_filtered = filtered_results[idx]
    
    println("\nFiltering efficiency:")
    println("  Best filtered: $(best_filtered.label)")
    println("    nadw = $(best_filtered.nadw)")
    println("    max error = $(round(best_filtered.max_err, sigdigits=3))")
    println("    speedup vs reference = $(round(nado_ref / best_filtered.nadw, digits=1))x")
    
    # Find depth-based with similar accuracy
    depth_results = filter(r -> !occursin("filtered", r.label), results)
    similar_depth = filter(r -> r.max_err <= best_filtered.max_err * 10, depth_results)
    if !isempty(similar_depth)
        idx2 = argmin([r.nadw for r in similar_depth])
        comparable = similar_depth[idx2]
        println("\n  Comparable depth-based: $(comparable.label)")
        println("    nadw = $(comparable.nadw)")
        println("    max error = $(round(comparable.max_err, sigdigits=3))")
        
        if comparable.nadw > best_filtered.nadw
            savings = (comparable.nadw - best_filtered.nadw) / comparable.nadw * 100
            println("\n  → Filtering saves $(round(savings, digits=1))% of ADWs for similar accuracy")
        end
    end
end

println("\n" * "="^60)
println("Comparison complete!")
println("="^60)
