"""
Analyze which ADW components grow large during HSEOM evolution
"""

using KaisouEOM
using KaisouEOM: icm2ifs, liouville_ket!, liouville_bra!, lsrk4!
using LinearAlgebra

println("Loading packages...")
using QFiND
using ProlateSpheroidalWaveFunctions

println("=" ^ 60)
println("ADW Growth Analysis")
println("=" ^ 60)

# Parameters (same as ohmic_pswf.jl)
s = 1.0
γc = 50.0
λ = 10.0
T = 300.0

sd = PowerLawExpSD(s, γc; reorgene=λ)

# PSWF expansion
ω_min = -150.0
ω_max = 200.0
n_terms = 15
T_pswf = 700.0
t_end = 100.0
dt = 0.25

sbeta = BosonicQNSD(sd, T)
f = w -> sbeta(w; scale=icm2ifs) / pi
pswfft = pswf_expansion_fourier(f, ω_min * icm2ifs, ω_max * icm2ifs, T_pswf, n_terms; pm=-1.0)

c_coeffs = pswfft.coeffs
D_matrix = compute_time_derivative_matrix(pswfft)

# System setup
ε = 0.0
Δ = 100.0
H = ComplexF64[ε/2 Δ; Δ -ε/2] * icm2ifs
V = ComplexF64[1 0; 0 -1]

expon = zeros(ComplexF64, n_terms)
coeff = c_coeffs
bath = BathExp(expon, coeff, V)
noise = NoiseExp(bath)

phi0 = zeros(ComplexF64, n_terms)
for k in 1:n_terms
    phi0[k] = pswfft.basis[k](0.0)
end

ndepth = 6
system = HSEOMSystem(H, noise, D_matrix, phi0, ndepth; hierarchy=:depth)

println("\nSystem Info:")
println("  nadw = $(system.nadw)")
println("  ndepth = $ndepth")
println("  nterms = $(system.nterms)")

# Initial condition
Pk = initial_adw(system, 1)
Pb = initial_adw(system, 1)

# Evolution and analysis
println("\n" * "=" ^ 60)
println("Tracking ADW evolution")
println("=" ^ 60)

global t = 0.0
global step = 0
check_interval = 40  # Every 10 fs

while t <= t_end
    global t, step
    if step % check_interval == 0
        # Compute norms for each ADW
        adw_norms_ket = [norm(Pk[:, n]) for n in 1:system.nadw]
        adw_norms_bra = [norm(Pb[:, n]) for n in 1:system.nadw]
        
        # Find top ADWs
        sorted_idx_ket = sortperm(adw_norms_ket, rev=true)
        
        println("\n--- t = $(round(t, digits=2)) fs ---")
        println("Max |Pk| = $(round(maximum(abs.(Pk)), digits=6)), Total norm ket = $(round(norm(Pk), digits=6))")
        
        println("\nTop 10 ADWs by norm (ket):")
        println("  Rank   ADW#    Norm       Index (n₁,n₂,...n_$(n_terms))  Depth")
        for rank in 1:min(10, system.nadw)
            n = sorted_idx_ket[rank]
            idx_vec = system.adw_idx[:, n]
            depth = sum(idx_vec)
            println("  $(lpad(rank, 4))  $(lpad(n, 5))   $(round(adw_norms_ket[n], digits=6))   $(idx_vec')  $depth")
        end
        
        # Show which mode indices are large
        mode_contributions = zeros(n_terms)
        for n in 1:system.nadw
            for k in 1:n_terms
                mode_contributions[k] += system.adw_idx[k, n] * adw_norms_ket[n]
            end
        end
        println("\nMode contributions (weighted by ADW norm):")
        top_modes = sortperm(mode_contributions, rev=true)[1:min(5, n_terms)]
        for k in top_modes
            println("  Mode $k: $(round(mode_contributions[k], digits=4))")
        end
    end
    
    # Time step
    lsrk4!(Pk, dt, liouville_ket_normalized!, system)
    lsrk4!(Pb, dt, liouville_bra_normalized!, system)
    t += dt
    step += 1
end

# Final analysis
println("\n" * "=" ^ 60)
println("Final Analysis: Which hierarchy indices grow?")
println("=" ^ 60)

adw_norms_ket = [norm(Pk[:, n]) for n in 1:system.nadw]
sorted_idx_ket = sortperm(adw_norms_ket, rev=true)

println("\nTop 20 ADWs at final time:")
for rank in 1:min(20, system.nadw)
    n = sorted_idx_ket[rank]
    idx_vec = system.adw_idx[:, n]
    depth = sum(idx_vec)
    # Show which modes are non-zero
    nonzero_modes = findall(idx_vec .> 0)
    mode_str = isempty(nonzero_modes) ? "root" : "modes $(nonzero_modes)"
    println("  Rank $(lpad(rank, 2)): ADW#$(lpad(n, 4)), norm=$(round(adw_norms_ket[n], digits=6)), depth=$depth, $mode_str")
end

# Analyze by depth
println("\n" * "=" ^ 60)
println("ADW norm sum by depth:")
println("=" ^ 60)
for d in 0:ndepth
    depth_sum = sum(adw_norms_ket[n] for n in 1:system.nadw if sum(system.adw_idx[:, n]) == d)
    n_adw_at_depth = count(n -> sum(system.adw_idx[:, n]) == d, 1:system.nadw)
    println("  Depth $d: total norm = $(round(depth_sum, digits=6)), # ADWs = $n_adw_at_depth")
end
