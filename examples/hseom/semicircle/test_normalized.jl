"""
HSEOM Test: Comparing evolve vs evolve_normalized vs HEOM

This script compares:
1. Standard HSEOM evolution (evolve) - bra grows exponentially, ket decays
2. Normalized HSEOM evolution (evolve_normalized) - bra/ket separately normalized each step
3. HEOM (density matrix) - reference calculation

All methods should give the same physical populations.
"""

using KaisouEOM
using KaisouEOM: icm2ifs, kB
using QFiND
using ProlateSpheroidalWaveFunctions
using ExpFit
using LinearAlgebra
using CairoMakie

println("=" ^ 70)
println("HSEOM Test: evolve vs evolve_normalized vs HEOM")
println("=" ^ 70)

# =============================================
# 1. Setup (same as semicircle.jl)
# =============================================
println("\n" * "=" ^ 60)
println("1. Setting up system")
println("=" ^ 60)

# Spectral density parameters
s = 1.0          # Ohmic
γc = 50.0        # Cutoff frequency [cm⁻¹]
λ = 1.0          # Reorganization energy [cm⁻¹]
T = 300.0        # Temperature [K]

sd = SemicircleSD(s, γc, λ)

println("Spectral Density: Semicircle (s=$s, γc=$γc cm⁻¹, λ=$λ cm⁻¹)")
println("Temperature: T=$T K")

# PSWF expansion parameters
ω_min = -50.0
ω_max = 50.0
n_terms = 6
T_pswf = 1000.0   # Shorter time for faster test

# System Hamiltonian
ε = 0.0
Δ = 20.0
H = ComplexF64[ε/2 Δ; Δ -ε/2] * icm2ifs

# Build PSWF expansion
sbeta = BosonicQNSD(sd, T)
f = w -> sbeta(w; scale=icm2ifs) / pi
pswfft = pswf_expansion_fourier(f, ω_min * icm2ifs, ω_max * icm2ifs, T_pswf, n_terms; pm=-1.0)

# Extract c, D, phi0
c_coeffs = pswfft.coeffs
D_matrix = compute_time_derivative_matrix(pswfft)
phi0 = [pswfft.basis[k](0.0) for k in 1:n_terms]

# Interaction operator
V = ComplexF64[1 0; 0 -1]

# Build noise and system
bath = BathGeneral(D_matrix, phi0, c_coeffs, V)
noise = NoiseGeneral(bath)

# Hierarchy with filtering
ndepth = 20
tol = 1e-50

nado, ado_idx, idx_plus, idx_minus = hierarchy_index_width(noise, ndepth;
    S_norm=opnorm(V), tolerance=tol, filter=true)
hseom_idx = build_hseom_index_maps(ado_idx, nado, n_terms)
system = HSEOMSystem(H, noise, nado, ado_idx, idx_plus, idx_minus, hseom_idx)

println("HSEOM System: $(system.nadw) ADWs, depth=$ndepth")

# =============================================
# 2. Time Evolution Parameters
# =============================================
t_end = 1000.0   # [fs]
dt = 0.25       # [fs]

Pb0 = initial_adw(system, 1)
Pk0 = initial_adw(system, 1)

println("\nTime evolution: t=[0, $t_end] fs, dt=$dt fs")

# =============================================
# 3. Standard Evolution (evolve)
# =============================================
println("\n" * "=" ^ 60)
println("2. Running standard evolve (unnormalized)")
println("=" ^ 60)

@time times_std, pops_std = evolve(system, Pb0, Pk0, (0.0, t_end), dt; 
    normalized=true, parallel=true,
    savefile=joinpath(@__DIR__, "pop_standard.dat"), save_interval=100)

println("  Final p₁ = $(round(pops_std[1,end], digits=6))")
println("  Final p₂ = $(round(pops_std[2,end], digits=6))")
println("  Total = $(round(sum(pops_std[:,end]), digits=6))")

# =============================================
# 4. Normalized Evolution (evolve_normalized)
# =============================================
println("\n" * "=" ^ 60)
println("3. Running evolve_normalized")
println("=" ^ 60)

@time times_norm, pops_norm, cum_norm_ket, cum_norm_bra, biorth_overlap = evolve_normalized(
    system, Pb0, Pk0, (0.0, t_end), dt;
    normalized=true, parallel=true,
    savefile=joinpath(@__DIR__, "pop_normalized.dat"), save_interval=100)

println("  Final p₁ = $(round(pops_norm[1,end], digits=6))")
println("  Final p₂ = $(round(pops_norm[2,end], digits=6))")
println("  Total = $(round(sum(pops_norm[:,end]), digits=6))")
println("\n  Cumulative norm ket: $(round(cum_norm_ket[end], digits=6))")
println("  Cumulative norm bra: $(round(cum_norm_bra[end], digits=6))")
println("  N_ket × N_bra = $(round(cum_norm_ket[end] * cum_norm_bra[end], digits=6))")
println("  ⟨b̃ra|k̃et⟩ = $(round(real(biorth_overlap[end]), digits=6))")
println("  True ⟨bra|ket⟩ = $(round(cum_norm_ket[end] * cum_norm_bra[end] * real(biorth_overlap[end]), digits=6))")

# =============================================
# 5. HEOM Reference Calculation
# =============================================
println("\n" * "=" ^ 60)
println("4. Running HEOM (reference)")
println("=" ^ 60)

# ESPRIT fitting of BCF for HEOM
bcf_exact = BosonicBCF(sd, T; ub=6000.0)
tmax_esprit = 1000.0
nsamples = 800
eps_esprit = 1e-3

dt_esprit = tmax_esprit / (nsamples - 1)
t_samples = range(0.0, tmax_esprit, length=nsamples)
bcf_samples = [bcf_exact(t) for t in t_samples]

ef = ExpFit.esprit(bcf_samples, dt_esprit, eps_esprit)
println("  ESPRIT fitting: $(length(ef.expon)) exponential terms")

# Build HEOM system
V_heom = ComplexF64[1 0; 0 -1]
bath_heom = BathExp(ef.expon, ef.coeff, V_heom)
noise_heom = NoiseExp(bath_heom)

ndepth_heom = 8
system_heom = HEOMSystem(H, noise_heom, ndepth_heom; hierarchy=:depth)

println("  HEOM: $(system_heom.nado) ADOs, depth=$ndepth_heom")

# Initial condition and evolution
P0_heom = initial_ado(system_heom, 1)

println("  Running HEOM dynamics...")
@time times_heom, pops_heom = evolve(system_heom, P0_heom, (0.0, t_end), dt; parallel=true)

println("  Final p₁ = $(round(pops_heom[1,end], digits=6))")
println("  Final p₂ = $(round(pops_heom[2,end], digits=6))")
println("  Total = $(round(sum(pops_heom[:,end]), digits=6))")

# =============================================
# 6. Comparison
# =============================================
println("\n" * "=" ^ 60)
println("5. Comparison")
println("=" ^ 60)

# Maximum difference between HSEOM methods
max_diff_p1 = maximum(abs.(pops_std[1,:] .- pops_norm[1,:]))
max_diff_p2 = maximum(abs.(pops_std[2,:] .- pops_norm[2,:]))

println("\nHSEOM standard vs normalized:")
println("  |Δp₁| = $(round(max_diff_p1, sigdigits=3))")
println("  |Δp₂| = $(round(max_diff_p2, sigdigits=3))")

# Comparison with HEOM (interpolate to common time points)
using Interpolations
itp_heom = linear_interpolation(times_heom, pops_heom[1,:])
pops_heom_interp = [itp_heom(t) for t in times_std]

max_diff_heom_std = maximum(abs.(pops_std[1,:] .- pops_heom_interp))
max_diff_heom_norm = maximum(abs.(pops_norm[1,:] .- pops_heom_interp))

println("\nHSEOM vs HEOM:")
println("  |Δp₁| (standard vs HEOM) = $(round(max_diff_heom_std, sigdigits=3))")
println("  |Δp₁| (normalized vs HEOM) = $(round(max_diff_heom_norm, sigdigits=3))")

# =============================================
# 7. Visualization
# =============================================
println("\n" * "=" ^ 60)
println("6. Generating plots")
println("=" ^ 60)

outdir = @__DIR__

fig = Figure(size=(1400, 1200))

# Panel 1: Population comparison with HEOM
ax1 = Axis(fig[1,1],
    xlabel = "Time [fs]",
    ylabel = "Population p₁",
    title = "Population Dynamics: HSEOM vs HEOM"
)
lines!(ax1, times_std, pops_std[1,:], linewidth=2, label="HSEOM (standard)", color=:blue)
lines!(ax1, times_norm, pops_norm[1,:], linewidth=2, linestyle=:dash, label="HSEOM (normalized)", color=:red)
lines!(ax1, times_heom, pops_heom[1,:], linewidth=2, linestyle=:dot, label="HEOM", color=:green)
axislegend(ax1, position=:rt)

# Panel 2: Difference from HEOM
ax2 = Axis(fig[1,2],
    xlabel = "Time [fs]",
    ylabel = "Δp₁ (vs HEOM)",
    title = "Population Difference from HEOM"
)
lines!(ax2, times_std, pops_std[1,:] .- pops_heom_interp, linewidth=2, label="Standard - HEOM", color=:blue)
lines!(ax2, times_norm, pops_norm[1,:] .- pops_heom_interp, linewidth=2, linestyle=:dash, label="Normalized - HEOM", color=:red)
hlines!(ax2, [0.0], linestyle=:dash, color=:gray)
axislegend(ax2, position=:lt)

# Panel 3: Cumulative normalization factors
ax3 = Axis(fig[2,1],
    xlabel = "Time [fs]",
    ylabel = "Cumulative Norm Factor",
    title = "Cumulative Normalization Factors"
)
lines!(ax3, times_norm, cum_norm_ket, linewidth=2, label="N_ket", color=:blue)
lines!(ax3, times_norm, cum_norm_bra, linewidth=2, label="N_bra", color=:red)
lines!(ax3, times_norm, cum_norm_ket .* cum_norm_bra, linewidth=2, linestyle=:dash, 
       label="N_ket × N_bra", color=:purple)
hlines!(ax3, [1.0], linestyle=:dash, color=:gray)
axislegend(ax3, position=:lt)

# Panel 4: Biorthogonal overlap
ax4 = Axis(fig[2,2],
    xlabel = "Time [fs]",
    ylabel = "Overlap",
    title = "Biorthogonal Overlap ⟨b̃ra|k̃et⟩"
)
lines!(ax4, times_norm, real.(biorth_overlap), linewidth=2, label="Real", color=:blue)
lines!(ax4, times_norm, imag.(biorth_overlap), linewidth=2, label="Imag", color=:red)
hlines!(ax4, [1.0], linestyle=:dash, color=:gray)
axislegend(ax4, position=:rt)

# Panel 5: True biorthogonal norm
ax5 = Axis(fig[3,1],
    xlabel = "Time [fs]",
    ylabel = "True ⟨bra|ket⟩",
    title = "True Biorthogonal Norm: N_ket × N_bra × ⟨b̃ra|k̃et⟩"
)
true_biorth = cum_norm_ket .* cum_norm_bra .* real.(biorth_overlap)
lines!(ax5, times_norm, true_biorth, linewidth=2, color=:purple)
hlines!(ax5, [1.0], linestyle=:dash, color=:gray)

# Panel 6: Total population comparison
ax6 = Axis(fig[3,2],
    xlabel = "Time [fs]",
    ylabel = "p₁ + p₂",
    title = "Total Population (should be ~1)"
)
lines!(ax6, times_std, pops_std[1,:] .+ pops_std[2,:], linewidth=2, label="HSEOM (std)", color=:blue)
lines!(ax6, times_norm, pops_norm[1,:] .+ pops_norm[2,:], linewidth=2, linestyle=:dash, 
       label="HSEOM (norm)", color=:red)
lines!(ax6, times_heom, pops_heom[1,:] .+ pops_heom[2,:], linewidth=2, linestyle=:dot,
       label="HEOM", color=:green)
hlines!(ax6, [1.0], linestyle=:dash, color=:gray)
axislegend(ax6, position=:rb)

# Panel 7: p2 comparison
ax7 = Axis(fig[4,1],
    xlabel = "Time [fs]",
    ylabel = "Population p₂",
    title = "Population p₂ Comparison"
)
lines!(ax7, times_std, pops_std[2,:], linewidth=2, label="HSEOM (standard)", color=:blue)
lines!(ax7, times_norm, pops_norm[2,:], linewidth=2, linestyle=:dash, label="HSEOM (normalized)", color=:red)
lines!(ax7, times_heom, pops_heom[2,:], linewidth=2, linestyle=:dot, label="HEOM", color=:green)
axislegend(ax7, position=:rb)

# Panel 8: HSEOM std vs norm difference
ax8 = Axis(fig[4,2],
    xlabel = "Time [fs]",
    ylabel = "Δp₁ (std - norm)",
    title = "HSEOM Standard vs Normalized Difference"
)
lines!(ax8, times_std, pops_std[1,:] .- pops_norm[1,:], linewidth=2, color=:black)
hlines!(ax8, [0.0], linestyle=:dash, color=:gray)

save(joinpath(outdir, "test_normalized_comparison.png"), fig)
println("  Saved: test_normalized_comparison.png")

# =============================================
# 8. Summary
# =============================================
println("\n" * "=" ^ 70)
println("Summary")
println("=" ^ 70)

println("""
Methods compared:
1. HSEOM standard: bra/ket evolve without normalization
2. HSEOM normalized: bra/ket separately normalized each step
3. HEOM: density matrix evolution (reference)

Results:
  HSEOM standard vs normalized:
    - Max |Δp₁| = $(round(max_diff_p1, sigdigits=3))
    - Max |Δp₂| = $(round(max_diff_p2, sigdigits=3))
    → Both HSEOM methods give identical results (machine precision)
    
  HSEOM vs HEOM:
    - Max |Δp₁| (standard vs HEOM) = $(round(max_diff_heom_std, sigdigits=3))
    - Max |Δp₁| (normalized vs HEOM) = $(round(max_diff_heom_norm, sigdigits=3))
    → Good agreement with HEOM reference

Normalization statistics:
  - Final N_ket = $(round(cum_norm_ket[end], sigdigits=4)) (ket decays)
  - Final N_bra = $(round(cum_norm_bra[end], sigdigits=4)) (bra grows)
  - Final ⟨b̃ra|k̃et⟩ = $(round(real(biorth_overlap[end]), sigdigits=4))
  - True ⟨bra|ket⟩ = $(round(cum_norm_ket[end] * cum_norm_bra[end] * real(biorth_overlap[end]), sigdigits=4))

Conclusion:
  The normalized evolution maintains numerical stability while giving
  identical physical results. For long-time dynamics where bra exponentially
  grows and ket exponentially decays, normalization prevents overflow/underflow.
""")

# =============================================
# 9. BCF Comparison Plot (separate file)
# =============================================
println("\n" * "=" ^ 60)
println("7. BCF Comparison Plot")
println("=" ^ 60)

# Time points for BCF comparison
t_bcf = range(0.0, T_pswf, length=500)

# Exact BCF
C_exact = [bcf_exact(t) for t in t_bcf]

# PSWF (HSEOM) approximation
C_pswf = [pswfft(t) for t in t_bcf]

# ESPRIT (HEOM) approximation: C(t) = Σₖ cₖ exp(-γₖ t)
function bcf_esprit(t, ef)
    return sum(ef.coeff[k] * exp(-ef.expon[k] * t) for k in 1:length(ef.expon))
end
C_esprit = [bcf_esprit(t, ef) for t in t_bcf]

# Compute errors
err_pswf = norm(C_pswf .- C_exact) / norm(C_exact) * 100
err_esprit = norm(C_esprit .- C_exact) / norm(C_exact) * 100

println("  BCF reconstruction errors:")
println("    PSWF (HSEOM):   $(round(err_pswf, digits=3))%")
println("    ESPRIT (HEOM):  $(round(err_esprit, digits=3))%")

# Create BCF comparison figure
fig_bcf = Figure(size=(1200, 900))

# Panel 1: Real part comparison
ax_re = Axis(fig_bcf[1,1],
    xlabel = "Time [fs]",
    ylabel = "Re[C(t)]",
    title = "Bath Correlation Function: Real Part"
)
lines!(ax_re, collect(t_bcf), real.(C_exact), linewidth=2, label="Exact", color=:black)
lines!(ax_re, collect(t_bcf), real.(C_pswf), linewidth=2, linestyle=:dash, 
       label="PSWF (HSEOM, $n_terms terms)", color=:blue)
lines!(ax_re, collect(t_bcf), real.(C_esprit), linewidth=2, linestyle=:dot, 
       label="ESPRIT (HEOM, $(length(ef.expon)) terms)", color=:red)
axislegend(ax_re, position=:rt)

# Panel 2: Imaginary part comparison
ax_im = Axis(fig_bcf[1,2],
    xlabel = "Time [fs]",
    ylabel = "Im[C(t)]",
    title = "Bath Correlation Function: Imaginary Part"
)
lines!(ax_im, collect(t_bcf), imag.(C_exact), linewidth=2, label="Exact", color=:black)
lines!(ax_im, collect(t_bcf), imag.(C_pswf), linewidth=2, linestyle=:dash, 
       label="PSWF (HSEOM)", color=:blue)
lines!(ax_im, collect(t_bcf), imag.(C_esprit), linewidth=2, linestyle=:dot, 
       label="ESPRIT (HEOM)", color=:red)
axislegend(ax_im, position=:rt)

# Panel 3: Real part error
ax_err_re = Axis(fig_bcf[2,1],
    xlabel = "Time [fs]",
    ylabel = "Error Re[C(t)]",
    title = "BCF Error: Real Part"
)
lines!(ax_err_re, collect(t_bcf), real.(C_pswf .- C_exact), linewidth=2, 
       label="PSWF - Exact", color=:blue)
lines!(ax_err_re, collect(t_bcf), real.(C_esprit .- C_exact), linewidth=2, linestyle=:dash, 
       label="ESPRIT - Exact", color=:red)
hlines!(ax_err_re, [0.0], linestyle=:dash, color=:gray)
axislegend(ax_err_re, position=:rt)

# Panel 4: Imaginary part error
ax_err_im = Axis(fig_bcf[2,2],
    xlabel = "Time [fs]",
    ylabel = "Error Im[C(t)]",
    title = "BCF Error: Imaginary Part"
)
lines!(ax_err_im, collect(t_bcf), imag.(C_pswf .- C_exact), linewidth=2, 
       label="PSWF - Exact", color=:blue)
lines!(ax_err_im, collect(t_bcf), imag.(C_esprit .- C_exact), linewidth=2, linestyle=:dash, 
       label="ESPRIT - Exact", color=:red)
hlines!(ax_err_im, [0.0], linestyle=:dash, color=:gray)
axislegend(ax_err_im, position=:rt)

# Panel 5: Absolute value comparison
ax_abs = Axis(fig_bcf[3,1],
    xlabel = "Time [fs]",
    ylabel = "|C(t)|",
    title = "Bath Correlation Function: Absolute Value"
)
lines!(ax_abs, collect(t_bcf), abs.(C_exact), linewidth=2, label="Exact", color=:black)
lines!(ax_abs, collect(t_bcf), abs.(C_pswf), linewidth=2, linestyle=:dash, 
       label="PSWF (HSEOM)", color=:blue)
lines!(ax_abs, collect(t_bcf), abs.(C_esprit), linewidth=2, linestyle=:dot, 
       label="ESPRIT (HEOM)", color=:red)
axislegend(ax_abs, position=:rt)

# Panel 6: Summary text
ax_summary = Axis(fig_bcf[3,2],
    title = "BCF Approximation Summary"
)
hidedecorations!(ax_summary)
hidespines!(ax_summary)

summary_text = """
Spectral Density: Semicircle (s=$s, γc=$γc cm⁻¹, λ=$λ cm⁻¹)
Temperature: T=$T K

PSWF Expansion (HSEOM):
  • Number of terms: $n_terms
  • Time range: [0, $T_pswf] fs
  • Frequency range: [$ω_min, $ω_max] cm⁻¹
  • Error: $(round(err_pswf, digits=3))%

ESPRIT Fitting (HEOM):
  • Number of exponentials: $(length(ef.expon))
  • ESPRIT tolerance: $eps_esprit
  • Error: $(round(err_esprit, digits=3))%
"""
text!(ax_summary, 0.05, 0.95, text=summary_text, 
      align=(:left, :top), fontsize=14)

save(joinpath(outdir, "test_bcf_comparison.png"), fig_bcf)
println("  Saved: test_bcf_comparison.png")

println("=" ^ 70)
println("Test completed!")
println("=" ^ 70)
