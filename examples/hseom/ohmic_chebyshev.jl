"""
HSEOM Example: Two-Level System with Ohmic Bath (Chebyshev Expansion)

This example demonstrates:
1. Chebyshev expansion of bath correlation function using QFiND
2. Construction of c (expansion coefficients) and D (derivative matrix)
3. HSEOM dynamics with bra/ket simultaneous evolution
4. Population dynamics visualization

For HSEOM, the BCF is expanded as:
    C(t) = Σₖ cₖ φₖ(t)
    ∂ₜφₖ = Σₗ Dₖₗ φₗ(t)

where φₖ(t) = Jₖ(Ω*t) * exp(-i*ω̄*t) (Bessel-Chebyshev basis).
"""

using KaisouEOM
using KaisouEOM: icm2ifs, kB
using QFiND
using LinearAlgebra
using CairoMakie

println("=" ^ 60)
println("HSEOM Example: Ohmic Bath with Chebyshev Expansion")
println("=" ^ 60)

# =============================================
# 1. Spectral Density Parameters
# =============================================
println("\n" * "=" ^ 60)
println("1. Setting up spectral density and Chebyshev expansion")
println("=" ^ 60)

# Spectral density: J(ω) = α * ω^s * exp(-ω/γc)
s = 1.0          # Ohmic (s=1), sub-Ohmic (s<1), super-Ohmic (s>1)
γc = 50.0        # Cutoff frequency [cm⁻¹]
λ = 10.0         # Reorganization energy [cm⁻¹] (smaller for stability)
T = 300.0        # Temperature [K]

sd = PowerLawExpSD(s, γc; reorgene=λ)
reorgene_calc = reorganization_energy(sd; ub=6000.0)

println("\nSpectral Density Parameters:")
println("  Type: Ohmic (s=$s)")
println("  Cutoff frequency γc = $γc cm⁻¹")
println("  Reorganization energy λ = $λ cm⁻¹ (calc: $(round(reorgene_calc, digits=2)))")
println("  Temperature T = $T K")

# =============================================
# 2. Chebyshev Expansion of BCF
# =============================================

# Chebyshev expansion parameters
ω_min = -700.0   # Lower frequency bound [cm⁻¹]
ω_max = 700.0    # Upper frequency bound [cm⁻¹]
n_terms = 20     # Number of Chebyshev terms

println("\nChebyshev Expansion Parameters:")
println("  Frequency range: [$ω_min, $ω_max] cm⁻¹")
println("  Number of terms: $n_terms")

# Perform Chebyshev expansion
cheb = chebyshev_expansion(sd, T, ω_min, ω_max, n_terms)

println("\nChebyshev Expansion Result:")
println("  ω̄ = $(cheb.ω_bar) (center frequency)")
println("  Ω = $(cheb.Ω) (half-width)")

# =============================================
# 3. Extract c and D for HSEOM
# =============================================
println("\n" * "=" ^ 60)
println("2. Extracting c (coefficients) and D (derivative matrix)")
println("=" ^ 60)

# Expansion coefficients c (already in 1/fs² units from QFiND)
c_coeffs = cheb.coeffs

# Derivative matrix D (scaled to 1/fs)
D_matrix = chebyshev_derivative_matrix(cheb)

println("\nExpansion coefficients c (first 5 terms):")
for k in 1:min(5, n_terms)
    println("  c[$k] = $(c_coeffs[k])")
end

println("\nDerivative matrix D (first 5x5 block):")
display(D_matrix[1:min(5, n_terms), 1:min(5, n_terms)])

# Verify BCF reconstruction
bcf_exact = BosonicBCF(sd, T; ub=6000.0)
bcf_cheb = chebyshev_bcf(cheb)

t_test = range(0.01, 200.0, length=100)
C_exact = [bcf_exact(t) for t in t_test]
C_cheb = [bcf_cheb(t) for t in t_test]
recon_error = norm(C_cheb .- C_exact) / norm(C_exact)

println("\nBCF Reconstruction Error: $(round(recon_error * 100, digits=4))%")

# =============================================
# 4. Build HSEOM System
# =============================================
println("\n" * "=" ^ 60)
println("3. Building HSEOM System")
println("=" ^ 60)

# System Hamiltonian (two-level system)
ε = 0.0      # Energy difference [cm⁻¹]
Δ = 100.0    # Tunneling coupling [cm⁻¹]
H = ComplexF64[ε/2 Δ; Δ -ε/2] * icm2ifs  # Convert to [1/fs]

println("\nSystem Hamiltonian:")
println("  Energy difference ε = $ε cm⁻¹")
println("  Tunneling coupling Δ = $Δ cm⁻¹")

# Interaction operator (σz coupling)
V = ComplexF64[1 0; 0 -1]

# Create Noise structure
# For HSEOM with Chebyshev: expon is not used directly, D matrix handles time evolution
# We use placeholder exponentials and the actual Chebyshev coefficients
expon = zeros(ComplexF64, n_terms)  # Placeholder (D matrix will be used instead)
coeff = c_coeffs  # Chebyshev expansion coefficients

bath = Bath(expon, coeff, V; add_conjugate=false)  # No conjugate for Chebyshev
noise = Noise(bath)

# HSEOM system with D matrix
ndepth = 6
system = HSEOMSystem(H, noise, D_matrix, ndepth; hierarchy=:depth)

println("\nHSEOM System:")
println("  Hierarchy depth: $ndepth")
println("  Number of ADWs: $(system.nadw)")
println("  Hilbert space dimension: $(system.ndim)")
println("  Number of BCF terms: $(system.nterms)")

# =============================================
# 5. Time Evolution
# =============================================
println("\n" * "=" ^ 60)
println("4. Running HSEOM Dynamics")
println("=" ^ 60)

# Initial condition: |1⟩ (localized on state 1)
Pb0 = initial_adw(system, 1)  # bra side
Pk0 = initial_adw(system, 1)  # ket side

# Time evolution parameters
t_end = 200.0    # [fs]
dt = 0.5         # [fs] (smaller for stability)

println("\nTime Evolution Parameters:")
println("  Initial state: |1⟩")
println("  Time range: [0, $t_end] fs")
println("  Time step: $dt fs")
println("  Total steps: $(Int(t_end/dt))")

println("\nRunning dynamics...")
@time times, pops = evolve(system, Pb0, Pk0, (0.0, t_end), dt)

println("\nResults:")
println("  Initial population: p₁=$(round(pops[1,1], digits=4)), p₂=$(round(pops[2,1], digits=4))")
println("  Final population:   p₁=$(round(pops[1,end], digits=4)), p₂=$(round(pops[2,end], digits=4))")
println("  Total population:   $(round(pops[1,end] + pops[2,end], digits=4))")

# =============================================
# 6. Visualization
# =============================================
println("\n" * "=" ^ 60)
println("5. Generating plots")
println("=" ^ 60)

outdir = @__DIR__

# Figure 1: Population dynamics
fig1 = Figure(size=(800, 500))
ax1 = Axis(fig1[1, 1],
    xlabel = "Time [fs]",
    ylabel = "Population",
    title = "HSEOM: Two-Level System with Ohmic Bath\n(λ=$λ cm⁻¹, γc=$γc cm⁻¹, T=$T K, $n_terms Chebyshev terms)"
)
lines!(ax1, times, pops[1, :], linewidth=2, label="p₁", color=:blue)
lines!(ax1, times, pops[2, :], linewidth=2, label="p₂", color=:red)
lines!(ax1, times, pops[1, :] .+ pops[2, :], linewidth=1, linestyle=:dash, 
       label="Total", color=:black)
axislegend(ax1, position=:rt)
save(joinpath(outdir, "hseom_population_ohmic.png"), fig1)
println("  Saved: hseom_population_ohmic.png")

# Figure 2: BCF comparison (Exact vs Chebyshev)
fig2 = Figure(size=(800, 500))
ax2 = Axis(fig2[1, 1],
    xlabel = "Time [fs]",
    ylabel = "C(t)",
    title = "Bath Correlation Function: Exact vs Chebyshev ($n_terms terms)\nReconstruction error: $(round(recon_error*100, digits=3))%"
)
lines!(ax2, collect(t_test), real.(C_exact), linewidth=2, label="Re[C(t)] Exact", color=:blue)
lines!(ax2, collect(t_test), real.(C_cheb), linewidth=2, linestyle=:dash, 
       label="Re[C(t)] Chebyshev", color=:cyan)
lines!(ax2, collect(t_test), imag.(C_exact), linewidth=2, label="Im[C(t)] Exact", color=:red)
lines!(ax2, collect(t_test), imag.(C_cheb), linewidth=2, linestyle=:dash, 
       label="Im[C(t)] Chebyshev", color=:orange)
axislegend(ax2, position=:rt)
save(joinpath(outdir, "hseom_bcf_chebyshev.png"), fig2)
println("  Saved: hseom_bcf_chebyshev.png")

# Figure 3: D matrix visualization
fig3 = Figure(size=(600, 500))
ax3 = Axis(fig3[1, 1],
    xlabel = "Column index",
    ylabel = "Row index",
    title = "Derivative Matrix D (absolute values)",
    yreversed = true
)
hm = heatmap!(ax3, abs.(D_matrix), colormap=:viridis)
Colorbar(fig3[1, 2], hm, label="|Dₖₗ|")
save(joinpath(outdir, "hseom_D_matrix.png"), fig3)
println("  Saved: hseom_D_matrix.png")

println("\n" * "=" ^ 60)
println("HSEOM Example completed!")
println("=" ^ 60)
