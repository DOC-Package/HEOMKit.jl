"""
HSEOM Example: Two-Level System with Ohmic Bath (PSWF Expansion)

This example demonstrates:
1. PSWF (Prolate Spheroidal Wave Function) expansion of bath correlation function
2. Construction of c (expansion coefficients) and D (derivative matrix)
3. HSEOM dynamics with bra/ket simultaneous evolution
4. Population dynamics visualization

For HSEOM, the BCF is expanded as:
    C(t) = Σₖ cₖ φₖ(t)
    ∂ₜφₖ = Σₗ Dₖₗ φₗ(t)

where φₖ(t) are PSWF basis functions with bandwidth-limited property.
"""

using KaisouEOM
using KaisouEOM: icm2ifs, kB
using QFiND
using ProlateSpheroidalWaveFunctions
using LinearAlgebra
using CairoMakie

println("=" ^ 60)
println("HSEOM Example: Ohmic Bath with PSWF Expansion")
println("=" ^ 60)

# =============================================
# 1. Spectral Density Parameters
# =============================================
println("\n" * "=" ^ 60)
println("1. Setting up spectral density and PSWF expansion")
println("=" ^ 60)

# Spectral density: J(ω) = α * ω^s * exp(-ω/γc)
s = 1.0          # Ohmic (s=1), sub-Ohmic (s<1), super-Ohmic (s>1)
γc = 50.0        # Cutoff frequency [cm⁻¹]
λ = 5.0         # Reorganization energy [cm⁻¹]
T = 300.0        # Temperature [K]

sd = PowerLawExpSD(s, γc; reorgene=λ)
reorgene_calc = reorganization_energy(sd; ub=6000.0)

println("\nSpectral Density Parameters:")
println("  Type: Ohmic (s=$s)")
println("  Cutoff frequency γc = $γc cm⁻¹")
println("  Reorganization energy λ = $λ cm⁻¹ (calc: $(round(reorgene_calc, digits=2)))")
println("  Temperature T = $T K")

# =============================================
# 2. PSWF Expansion of BCF
# =============================================

# PSWF expansion parameters
ω_min = -150.0   # Lower frequency bound [cm⁻¹]
ω_max = 200.0    # Upper frequency bound [cm⁻¹]
n_terms = 15     # Number of PSWF terms
T_pswf = 700.0  # Time duration parameter [fs]
# Time evolution parameters
t_end = 700.0    # [fs]
dt = 0.25         # [fs]

println("\nPSWF Expansion Parameters:")
println("  Frequency range: [$ω_min, $ω_max] cm⁻¹")
println("  Number of terms: $n_terms")
println("  Time duration: $T_pswf fs")

# Construct the quantum noise spectral density for PSWF expansion
sbeta = BosonicQNSD(sd, T)
f = w -> sbeta(w; scale=icm2ifs) / pi

# Perform PSWF expansion
pswfft = pswf_expansion_fourier(f, ω_min * icm2ifs, ω_max * icm2ifs, T_pswf, n_terms; pm=-1.0)

println("\nPSWF Expansion Result:")
println("  Bandwidth-limited: [$ω_min, $ω_max] cm⁻¹")
println("  Number of coefficients: $(length(pswfft.coeffs))")

# =============================================
# 3. Extract c and D for HSEOM
# =============================================
println("\n" * "=" ^ 60)
println("2. Extracting c (coefficients) and D (derivative matrix)")
println("=" ^ 60)

# Expansion coefficients c (already in appropriate units)
c_coeffs = pswfft.coeffs

# Derivative matrix D for PSWF basis
D_matrix = compute_time_derivative_matrix(pswfft)

println("\nExpansion coefficients c (first 5 terms):")
for k in 1:min(5, n_terms)
    println("  c[$k] = $(c_coeffs[k])")
end

println("\nDerivative matrix D (first 5x5 block):")
display(D_matrix[1:min(5, n_terms), 1:min(5, n_terms)])

# Verify BCF reconstruction
bcf_exact = BosonicBCF(sd, T; ub=6000.0)

t_test = range(0.01, T_pswf, length=500)
C_exact = [bcf_exact(t) for t in t_test]
C_pswf = [pswfft(t) for t in t_test]
recon_error = norm(C_pswf .- C_exact) / norm(C_exact)

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
# For HSEOM with PSWF: expon is not used directly, D matrix handles time evolution
expon = zeros(ComplexF64, n_terms)  # Placeholder (D matrix will be used instead)
coeff = c_coeffs  # PSWF expansion coefficients

bath = Bath(expon, coeff, V; add_conjugate=false)  # No conjugate for PSWF
noise = Noise(bath)

# φₖ(0) for PSWF expansion: evaluate basis functions at t=0
phi0 = zeros(ComplexF64, n_terms)
for k in 1:n_terms
    phi0[k] = pswfft.basis[k](0.0)
end

println("\nPSWF basis at t=0 (first 5 terms):")
for k in 1:min(5, n_terms)
    println("  φ[$k](0) = $(phi0[k])")
end

# HSEOM system with D matrix
ndepth = 6  # Keep small to avoid memory issues
system = HSEOMSystem(H, noise, D_matrix, phi0, ndepth; hierarchy=:depth)

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

println("\nTime Evolution Parameters:")
println("  Initial state: |1⟩")
println("  Time range: [0, $t_end] fs")
println("  Time step: $dt fs")
println("  Total steps: $(Int(t_end/dt))")

println("\nRunning dynamics...")
@time times, pops = evolve(system, Pb0, Pk0, (0.0, t_end), dt; normalized=true, parallel=true, savefile="pop.dat", save_interval=10)

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
    title = "HSEOM (PSWF): Two-Level System with Ohmic Bath\n(λ=$λ cm⁻¹, γc=$γc cm⁻¹, T=$T K, $n_terms PSWF terms)"
)
lines!(ax1, times, pops[1, :], linewidth=2, label="p₁", color=:blue)
lines!(ax1, times, pops[2, :], linewidth=2, label="p₂", color=:red)
lines!(ax1, times, pops[1, :] .+ pops[2, :], linewidth=1, linestyle=:dash, 
       label="Total", color=:black)
axislegend(ax1, position=:rt)
save(joinpath(outdir, "hseom_population_ohmic_pswf.png"), fig1)
println("  Saved: hseom_population_ohmic_pswf.png")

# Figure 2: BCF comparison (Exact vs PSWF)
fig2 = Figure(size=(800, 500))
ax2 = Axis(fig2[1, 1],
    xlabel = "Time [fs]",
    ylabel = "C(t)",
    title = "Bath Correlation Function: Exact vs PSWF ($n_terms terms)\nReconstruction error: $(round(recon_error*100, digits=3))%"
)
lines!(ax2, collect(t_test), real.(C_exact), linewidth=2, label="Re[C(t)] Exact", color=:blue)
lines!(ax2, collect(t_test), real.(C_pswf), linewidth=2, linestyle=:dash, 
       label="Re[C(t)] PSWF", color=:cyan)
lines!(ax2, collect(t_test), imag.(C_exact), linewidth=2, label="Im[C(t)] Exact", color=:red)
lines!(ax2, collect(t_test), imag.(C_pswf), linewidth=2, linestyle=:dash, 
       label="Im[C(t)] PSWF", color=:orange)
axislegend(ax2, position=:rt)
save(joinpath(outdir, "hseom_bcf_pswf.png"), fig2)
println("  Saved: hseom_bcf_pswf.png")

# Figure 3: D matrix visualization
fig3 = Figure(size=(600, 500))
ax3 = Axis(fig3[1, 1],
    xlabel = "Column index",
    ylabel = "Row index",
    title = "PSWF Derivative Matrix D (absolute values)",
    yreversed = true
)
hm = heatmap!(ax3, abs.(D_matrix), colormap=:viridis)
Colorbar(fig3[1, 2], hm, label="|Dₖₗ|")
save(joinpath(outdir, "hseom_D_matrix_pswf.png"), fig3)
println("  Saved: hseom_D_matrix_pswf.png")

# Figure 4: PSWF basis functions at t=0
fig4 = Figure(size=(600, 400))
ax4 = Axis(fig4[1, 1],
    xlabel = "Index k",
    ylabel = "φₖ(0)",
    title = "PSWF Basis Functions at t=0"
)
barplot!(ax4, 1:n_terms, real.(phi0), color=:blue, label="Real")
barplot!(ax4, 1:n_terms, imag.(phi0), color=:red, label="Imag", dodge=2)
axislegend(ax4)
save(joinpath(outdir, "hseom_phi0_pswf.png"), fig4)
println("  Saved: hseom_phi0_pswf.png")

println("\n" * "=" ^ 60)
println("HSEOM (PSWF) Example completed!")
println("=" ^ 60)
