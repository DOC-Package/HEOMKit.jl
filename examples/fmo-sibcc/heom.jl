using KaisouEOM
using KaisouEOM: icm2ifs
using QFiND
using ExpFit
using LinearAlgebra
using Serialization
using CairoMakie

function site_projector(d::Int, site::Int)
    S = zeros(ComplexF64, d, d)
    S[site, site] = 1.0
    return S
end

function interaction_picture_operator(H::AbstractMatrix, S::AbstractMatrix, tau::Real)
    U = exp(-1im * H * tau)
    return U * S * U'
end

function fit_error(reference, fit)
    return norm(fit .- reference) / norm(reference)
end

Temp = 300.0
ub = 500.0
E_reorg = 35.0

# SIBC/group-OLS sampling. Times are in fs.
tmin = 0.0
tmax = 600.0
nsamples = 180
n_output = 240
candidate_order = 20
selection_tolerance = 5e-2
max_selected_terms = 10

# HEOM time-evolution parameters.
t_end = 500.0
dt_evolve = 0.5
ndepth = 3
hierarchy_tolerance = 1e-5

outdir = @__DIR__
mkpath(outdir)

rfile = normpath(joinpath(@__DIR__, "..", "..", "..", "QFiND", "examples", "FMO", "r_fmo.bin"))
r = open(rfile, "r") do io
    deserialize(io)
end

sdens = AAAfittedSD(r, E_reorg)
println("FMO SIBC-HEOM example")
println("  Reorganization energy: $(sdens.reorgene) cm^-1")
println("  Temperature: $Temp K")
println("  Candidate ESPRIT order: $candidate_order")
println("  Group-OLS tolerance: $selection_tolerance")
println("  Maximum selected terms per site: $max_selected_terms")
bcf = BosonicBCF(sdens, Temp; ub=ub, rtol=1e-6)

# Three-site FMO Hamiltonian used in examples/fmo/heom.jl.
H = ComplexF64[
    310.0 -97.9   5.5
    -97.9 230.0  30.1
      5.5  30.1   0.0
] * icm2ifs

t_samples = collect(range(tmin, tmax, length=nsamples))
dt_sample = t_samples[2] - t_samples[1]
bcf_samples = ComplexF64[bcf(t) for t in t_samples]
candidate_fit = ExpFit.esprit(bcf_samples, dt_sample, candidate_order)
candidate_poles = ComplexF64.(candidate_fit.expon)

d = size(H, 1)
baths = BathExp[]
selected_bcf = Vector{Vector{ComplexF64}}(undef, d)

println()
println("System-weighted group-OLS pole selection")
for site in 1:d
    S = site_projector(d, site)
    T = [interaction_picture_operator(H, S, t) for t in t_samples]
    G = sibcc_gram(S, T, dt_sample, n_output)
    selected = select_exponentials_group_ols(
        G,
        bcf_samples,
        t_samples,
        candidate_poles;
        tolerance=selection_tolerance,
        max_terms=max_selected_terms,
    )
    selected_bcf[site] = selected.fitted_bcf
    bcf_error = fit_error(bcf_samples, selected.fitted_bcf)

    push!(baths, BathExp(selected.selected_poles, selected.coefficients, S))

    println("  Site $site")
    println("    Selected terms: $(length(selected.selected_indices))")
    println("    Selected candidate indices: $(selected.selected_indices)")
    println("    SIBC relative error: $(selected.relative_error)")
    println("    Euclidean BCF error: $bcf_error")
end

noise = NoiseExp(baths)
println()
println("HEOM noise")
println("  Baths: $(noise.nbath)")
println("  Expanded HEOM terms: $(noise.nterms)")
println("  Terms per bath after conjugate expansion: $(noise.nterms_bath)")

system = HEOMSystem(H, noise, ndepth;
    hierarchy=:width,
    tolerance=hierarchy_tolerance,
    filter=true,
)
println()
println("HEOM system")
println("  Hierarchy depth: $ndepth")
println("  ADOs: $(system.nado)")

P0 = initial_ado(system, 1)
popfile = joinpath(outdir, "pop_heom_sibcc_group_ols.dat")
println()
println("Running HEOM dynamics...")
times, pops = evolve(
    system,
    P0,
    (0.0, t_end),
    dt_evolve;
    parallel=true,
    savefile=popfile,
    save_interval=10,
)
println("  Done")
println("  Saved populations: $popfile")
println("  Final populations: rho11=$(pops[1,end]), rho22=$(pops[2,end]), rho33=$(pops[3,end])")

println()
println("Generating plots...")

fig1 = Figure(size=(800, 500))
ax1 = Axis(fig1[1, 1],
    xlabel="Time [fs]",
    ylabel="Population",
    title="FMO HEOM with SIBC group-OLS BCFs",
)
lines!(ax1, times, pops[1, :], linewidth=2, label="rho11", color=:blue)
lines!(ax1, times, pops[2, :], linewidth=2, label="rho22", color=:red)
lines!(ax1, times, pops[3, :], linewidth=2, label="rho33", color=:green)
axislegend(ax1, position=:rt)
pop_png = joinpath(outdir, "population_sibcc_group_ols.png")
save(pop_png, fig1)
println("  Saved: $pop_png")

for site in 1:d
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1],
        xlabel="Time [fs]",
        ylabel="C(t)",
        title="Site $site BCF: reference and SIBC group-OLS",
    )
    lines!(ax, t_samples, real.(bcf_samples), linewidth=3, linestyle=:dot, label="Re reference", color=:black)
    lines!(ax, t_samples, imag.(bcf_samples), linewidth=3, linestyle=:dot, label="Im reference", color=:gray35)
    lines!(ax, t_samples, real.(selected_bcf[site]), linewidth=2, label="Re group-OLS", color=:royalblue)
    lines!(ax, t_samples, imag.(selected_bcf[site]), linewidth=2, label="Im group-OLS", color=:orangered)
    axislegend(ax, position=:rt)
    bcf_png = joinpath(outdir, "bcf_sibcc_group_ols_site$(site).png")
    save(bcf_png, fig)
    println("  Saved: $bcf_png")
end

println()
println("Example completed")
