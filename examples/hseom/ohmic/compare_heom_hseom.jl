"""
HEOM vs HSEOM Comparison Test

Test using a manually defined Bessel sum BCF:
- HSEOM: Direct Bessel expansion C(t) = Σₖ cₖ Jₖ(Ωt)
- HEOM: Exponential fit of the same BCF

This validates HSEOM implementation by comparing with HEOM.
"""

using KaisouEOM
using KaisouEOM: icm2ifs
using SpecialFunctions: besselj
using ExpFit
using LinearAlgebra
using Statistics
using CairoMakie

println("=" ^ 60)
println("HEOM vs HSEOM Comparison Test")
println("(Bessel BCF vs Exponential Fit)")
println("=" ^ 60)

# =============================================
# Parameters
# =============================================
NL = 2           # 2-level system
Δ = 100.0        # Tunneling [cm⁻¹]

# Bessel BCF parameters
n_bessel = 5     # Number of Bessel terms
Ω = 0.01         # Bessel argument scale [1/fs]

# HEOM fit parameters
n_exp = 4        # Exponential terms for HEOM fit

println("\nParameters:")
println("  NL = $NL")
println("  Δ = $Δ cm⁻¹")
println("  n_bessel (HSEOM) = $n_bessel")
println("  Ω = $Ω [1/fs]")
println("  n_exp (HEOM fit) = $n_exp")

# =============================================
# Define Bessel BCF: C(t) = Σₖ cₖ Jₖ(Ωt)
# =============================================
println("\n" * "=" ^ 60)
println("1. Defining Bessel BCF")
println("=" ^ 60)

# Simple coefficients (make them decay with k for physical reasonableness)
c_bessel = ComplexF64[
    0.0003 + 0.0000im,    # c₀
    0.0000 - 0.0002im,   # c₁
    0.0003 + 0.0000im,   # c₂
    0.0000 - 0.00005im,  # c₃
    0.0001 + 0.0000im,  # c₄
]

# BCF function
function bcf_bessel(t)
    C = zero(ComplexF64)
    for k in 1:n_bessel
        C += c_bessel[k] * besselj(k-1, Ω * t)  # J_{k-1}(Ωt), k=1→J₀
    end
    return C
end

println("\nBessel coefficients cₖ:")
for k in 1:n_bessel
    println("  c[$k] = $(c_bessel[k])")
end

# D matrix for Bessel expansion: ∂ₜJₖ(Ωt) = Ω/2 * (Jₖ₋₁ - Jₖ₊₁)
# So D is tridiagonal with D[k,k-1] = +Ω/2, D[k,k+1] = -Ω/2
D_bessel = zeros(ComplexF64, n_bessel, n_bessel)
for k in 1:n_bessel
    if k > 1
        D_bessel[k, k-1] = Ω / 2
    end
    if k < n_bessel
        D_bessel[k, k+1] = -Ω / 2
    end
end
# Special case for k=0 (first row): ∂ₜJ₀ = -Ω*J₁
D_bessel[1, 2] = -Ω

println("\nD matrix (Bessel):")
display(D_bessel)

# φₖ(0) = Jₖ(0) = δₖ₀
phi0_bessel = zeros(ComplexF64, n_bessel)
phi0_bessel[1] = 1.0  # J₀(0) = 1

println("\nphi0 (Bessel initial values):")
println("  ", phi0_bessel)

# Test BCF at t=0
println("\nBCF at t=0: C(0) = $(bcf_bessel(0.0))")
println("  (should equal c[1] since J₀(0)=1, Jₖ(0)=0 for k>0)")

# =============================================
# HEOM: Exponential fit of Bessel BCF
# =============================================
println("\n" * "=" ^ 60)
println("2. Exponential Fit for HEOM")
println("=" ^ 60)

# Fit BCF with exponentials
t_fit_max = 300.0
nsamples = 300

# Fit real and imaginary parts separately
bcf_re(t) = real(bcf_bessel(t))
bcf_im(t) = imag(bcf_bessel(t))

fit = expfit(bcf_bessel, 0.01, t_fit_max, nsamples, n_exp)
#fit_im = expfit(bcf_im, 0.01, t_fit_max, nsamples, n_exp)

# Combine into HEOM format
γ_heom = ComplexF64[]
c_heom = ComplexF64[]

for i in 1:length(fit.coeff)
    push!(γ_heom, fit.expon[i])
    push!(c_heom, fit.coeff[i])
end
#for i in 1:length(fit_im.coeff)
#    push!(γ_heom, fit_im.expon[i])
#    push!(c_heom, fit_im.coeff[i] * 1.0im)
#end

nterms_heom = length(γ_heom)

println("\nExponential fit results:")
println("  Number of terms: $nterms_heom")
for k in 1:nterms_heom
    println("    γ[$k] = $(round(γ_heom[k], sigdigits=4)), c[$k] = $(round(c_heom[k], sigdigits=4))")
end

# Check fit quality
function bcf_exp(t)
    C = zero(ComplexF64)
    for k in 1:nterms_heom
        C += c_heom[k] * exp(-γ_heom[k] * t)
    end
    return C
end

t_test = range(0.01, 200.0, length=100)
C_bessel = [bcf_bessel(t) for t in t_test]
C_exp = [bcf_exp(t) for t in t_test]
fit_error = norm(C_exp .- C_bessel) / norm(C_bessel)

println("\nBCF fit error: $(round(fit_error * 100, digits=3))%")

# =============================================
# System Setup
# =============================================
println("\n" * "=" ^ 60)
println("3. Building Systems")
println("=" ^ 60)

H = ComplexF64[0 Δ; Δ 0] * icm2ifs  # [1/fs]
V = ComplexF64[1 0; 0 -1]           # σz coupling

ndepth = 10

# ----- HSEOM Setup -----
println("\nHSEOM Setup:")
expon_hseom = zeros(ComplexF64, n_bessel)  # Placeholder (D matrix used instead)
bath_hseom = Bath(expon_hseom, c_bessel, V; add_conjugate=false)
noise_hseom = Noise(bath_hseom)

hseom_system = HSEOMSystem(H, noise_hseom, D_bessel, ndepth; phi0=phi0_bessel, hierarchy=:depth)
println("  HSEOM ADWs: $(hseom_system.nadw)")
println("  HSEOM nterms: $(hseom_system.nterms)")

# Initial state: |1⟩
Pb0 = initial_adw(hseom_system, 1)
Pk0 = initial_adw(hseom_system, 1)

# ----- HEOM Setup -----
println("\nHEOM Setup:")
bath_heom = Bath(γ_heom, c_heom, V; add_conjugate=true)
noise_heom = Noise(bath_heom)

heom_system = HEOMSystem(H, noise_heom, ndepth; hierarchy=:depth)
println("  HEOM ADOs: $(heom_system.nado)")
println("  HEOM nterms: $(noise_heom.nterms)")

# Initial density matrix: |1⟩⟨1|
ρ0 = zeros(ComplexF64, NL, NL)
ρ0[1, 1] = 1.0
ρ0_heom = initial_ado(heom_system, ρ0)

# =============================================
# Time Evolution
# =============================================
println("\n" * "=" ^ 60)
println("4. Running Time Evolution")
println("=" ^ 60)

t_end = 200.0  # fs
dt = 0.5       # fs

println("\n  Time range: [0, $t_end] fs")
println("  Time step: $dt fs")

# HSEOM evolution
println("\nRunning HSEOM...")
@time times_hseom, pop_hseom = evolve(hseom_system, Pb0, Pk0, (0.0, t_end), dt)

# HEOM evolution
println("\nRunning HEOM...")
@time times_heom, pop_heom = evolve(heom_system, ρ0_heom, (0.0, t_end), dt)

# =============================================
# Comparison
# =============================================
println("\n" * "=" ^ 60)
println("5. Results Comparison")
println("=" ^ 60)

println("\nHSEOM Results (Bessel expansion):")
println("  Initial: p₁=$(round(pop_hseom[1,1], digits=4)), p₂=$(round(pop_hseom[2,1], digits=4))")
println("  Final:   p₁=$(round(pop_hseom[1,end], digits=4)), p₂=$(round(pop_hseom[2,end], digits=4))")
println("  Total:   $(round(pop_hseom[1,end] + pop_hseom[2,end], digits=6))")

println("\nHEOM Results (Exponential fit):")
println("  Initial: p₁=$(round(pop_heom[1,1], digits=4)), p₂=$(round(pop_heom[2,1], digits=4))")
println("  Final:   p₁=$(round(pop_heom[1,end], digits=4)), p₂=$(round(pop_heom[2,end], digits=4))")
println("  Total:   $(round(pop_heom[1,end] + pop_heom[2,end], digits=6))")

# Compute difference
diff_p1 = abs.(pop_heom[1, :] .- pop_hseom[1, :])
diff_p2 = abs.(pop_heom[2, :] .- pop_hseom[2, :])
max_diff = max(maximum(diff_p1), maximum(diff_p2))
mean_diff = (mean(diff_p1) + mean(diff_p2)) / 2

println("\nDifference (HEOM - HSEOM):")
println("  Max |Δp|:  $(round(max_diff, digits=4))")
println("  Mean |Δp|: $(round(mean_diff, digits=4))")
println("  BCF fit error: $(round(fit_error * 100, digits=3))%")

if max_diff < 0.05
    println("\n✓ HEOM and HSEOM results are consistent!")
else
    println("\n✗ Results differ - difference may be due to BCF fit error")
end

# =============================================
# Plots
# =============================================
println("\n" * "=" ^ 60)
println("6. Generating plots")
println("=" ^ 60)

fig = Figure(size=(900, 800))

# Population comparison
ax1 = Axis(fig[1, 1],
    xlabel = "Time [fs]",
    ylabel = "Population",
    title = "HEOM vs HSEOM: Population Dynamics\n(Bessel BCF, n=$n_bessel terms)"
)

lines!(ax1, times_hseom, pop_hseom[1, :], linewidth=3, label="p₁ (HSEOM)", color=:blue)
lines!(ax1, times_hseom, pop_hseom[2, :], linewidth=3, label="p₂ (HSEOM)", color=:red)
lines!(ax1, times_heom, pop_heom[1, :], linewidth=2, linestyle=:dash, 
       label="p₁ (HEOM)", color=:cyan)
lines!(ax1, times_heom, pop_heom[2, :], linewidth=2, linestyle=:dash, 
       label="p₂ (HEOM)", color=:orange)
axislegend(ax1, position=:rt)

# Difference plot
ax2 = Axis(fig[2, 1],
    xlabel = "Time [fs]",
    ylabel = "|HEOM - HSEOM|",
    title = "Population Difference"
)

lines!(ax2, times_heom, diff_p1, linewidth=2, label="|Δp₁|", color=:blue)
lines!(ax2, times_heom, diff_p2, linewidth=2, label="|Δp₂|", color=:red)
axislegend(ax2, position=:rt)

# BCF comparison
ax3 = Axis(fig[3, 1],
    xlabel = "Time [fs]",
    ylabel = "C(t)",
    title = "BCF: Bessel vs Exponential Fit (error: $(round(fit_error*100,digits=2))%)"
)

lines!(ax3, collect(t_test), real.(C_bessel), linewidth=2, label="Re[C] Bessel", color=:blue)
lines!(ax3, collect(t_test), real.(C_exp), linewidth=2, linestyle=:dash, 
       label="Re[C] ExpFit", color=:cyan)
lines!(ax3, collect(t_test), imag.(C_bessel), linewidth=2, label="Im[C] Bessel", color=:red)
lines!(ax3, collect(t_test), imag.(C_exp), linewidth=2, linestyle=:dash, 
       label="Im[C] ExpFit", color=:orange)
axislegend(ax3, position=:rt)

outdir = @__DIR__
save(joinpath(outdir, "compare_heom_hseom.png"), fig)
println("  Saved: compare_heom_hseom.png")

println("\n" * "=" ^ 60)
println("Comparison Test Completed!")
println("=" ^ 60)
