using KaisouEOM
using KaisouEOM: icm2ifs, kB
using QFiND
using ExpFit
using LinearAlgebra
using CairoMakie

# Spectral density parameters
s = 1.0        # Ohmic (s=1), sub-Ohmic (s<1), super-Ohmic (s>1)
γ = 50.0      # Cutoff frequency [cm⁻¹]
λ = 1.0       # Reorganization energy [cm⁻¹]
T = 300.0      # Temperature [K]
sd = PowerLawExpSD(s, γ; reorgene=λ)
bcf = BosonicBCF(sd, T)

# ESPRIT Fitting of Bath Correlation Function
# Sampling parameters
tmin = 0.0
tmax = 3000.0   # [fs]
nsamples = 500
eps = 1e-3     # ESPRIT tolerance
# Time evolution parameters
t_end = 3000.0    # [fs]
dt_evolve = 0.5  # [fs]

# System Hamiltonian (two-level system)
ε = 0.0      # Energy difference [cm⁻¹]
Δ = 20.0    # Tunneling coupling [cm⁻¹]
H = [-ε/2 -Δ/2; -Δ/2 ε/2] * icm2ifs  # Convert to [1/fs]

dt = (tmax - tmin) / (nsamples - 1)
t_samples = range(tmin, tmax, length=nsamples)
bcf_samples = [bcf(t) for t in t_samples]
# ESPRIT fitting
ef = ExpFit.esprit(bcf_samples, dt, eps)
println("\nESPRIT Fitting Result:")
println("  Number of exponential terms: $(length(ef.expon))")
# Check fitting accuracy
bcf_fit = [ef(t) for t in t_samples]
fit_error = norm(bcf_fit .- bcf_samples) / norm(bcf_samples)
println("  Relative fitting error: $(fit_error)")
# Display fitted parameters
println("\nFitted Bath Parameters (γₖ, cₖ):")
for i in 1:length(ef.expon)
    println("  k=$i: γ = $(ef.expon[i]), c = $(ef.coeff[i])")
end

# Build HEOM System
# Bath construction from ESPRIT results
expon = ef.expon
coeff = ef.coeff
V = ComplexF64[1 0; 0 -1]  # σz coupling
bath = BathExp(expon, coeff, V)
noise = NoiseExp(bath)

println("\nSystem Hamiltonian:")
println("  Energy difference ε = $ε cm⁻¹")
println("  Tunneling coupling Δ = $Δ cm⁻¹")

# HEOM system
ndepth = 6
system = HEOMSystem(H, noise, ndepth; hierarchy=:depth)
println("\nHEOM System:")
println("  Hierarchy depth: $ndepth")
println("  Number of ADOs: $(system.nado)")

# Initial condition: localized on state |1⟩
P0 = initial_ado(system, 1)

println("\nTime Evolution:")
println("  Initial state: |1⟩⟨1|")
println("  Time range: [0, $t_end] fs")
println("  Time step: $dt_evolve fs")

# Run dynamics
println("\nRunning HEOM dynamics...")
times, pops = evolve(system, P0, (0.0, t_end), dt_evolve; parallel=true, savefile="pop_heom.dat", save_interval=10)

println("  Done!")
println("  Final populations: ρ₁₁ = $(pops[1,end]), ρ₂₂ = $(pops[2,end])")

# =============================================
# 5. Visualization
# =============================================

println("\n" * "=" ^ 60)
println("Generating plots...")
println("=" ^ 60)

# Output directory
outdir = @__DIR__

# Figure 1: Population dynamics
fig1 = Figure(size=(800, 500))
ax1 = Axis(fig1[1, 1],
    xlabel = "Time [fs]",
    ylabel = "Population",
    title = "Two-Level System with Ohmic Bath\n(λ=$λ cm⁻¹, γ=$γ cm⁻¹, T=$T K)"
)
lines!(ax1, times, real.(pops[1, :]), linewidth=2, label="ρ₁₁", color=:blue)
lines!(ax1, times, real.(pops[2, :]), linewidth=2, label="ρ₂₂", color=:red)
axislegend(ax1, position=:rt)
save(joinpath(outdir, "population_ohmic.png"), fig1)
println("  Saved: $(joinpath(outdir, "population_ohmic.png"))")

# Figure 2: ESPRIT fitting quality (Real and Imaginary parts in one panel)
fig2 = Figure(size=(800, 500))
ax2 = Axis(fig2[1, 1],
    xlabel = "Time [fs]",
    ylabel = "C(t)",
    title = "BCF: Original vs ESPRIT Fit ($(length(ef.expon)) terms, error = $(round(fit_error, sigdigits=2)))"
)
lines!(ax2, collect(t_samples), real.(bcf_fit), linewidth=2, linestyle=:solid, label="Re[C(t)] ESPRIT", color=:blue)
lines!(ax2, collect(t_samples), real.(bcf_samples), linewidth=3, linestyle=:dot, label="Re[C(t)] Original", color=:black)
lines!(ax2, collect(t_samples), imag.(bcf_fit), linewidth=2, linestyle=:solid, label="Im[C(t)] ESPRIT", color=:red)
lines!(ax2, collect(t_samples), imag.(bcf_samples), linewidth=3, linestyle=:dot, label="Im[C(t)] Original", color=:black)
axislegend(ax2, position=:rt)

save(joinpath(outdir, "bcf_esprit_fit.png"), fig2)
println("  Saved: $(joinpath(outdir, "bcf_esprit_fit.png"))")

println("\n" * "=" ^ 60)
println("Example completed!")
println("=" ^ 60)
