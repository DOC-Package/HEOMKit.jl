using KaisouEOM
using KaisouEOM: icm2ifs
using QFiND
using DelimitedFiles
using ExpFit
using LinearAlgebra

const TEMPERATURE = 300.0

# SIBCC sampling parameters. The output-time grid uses the same spacing as tau.
const T_C = 500.0
const N_TAU = 251
const N_OUTPUT_TIME = 251

# ESPRIT candidate-generation grid. This can be chosen independently of the
# SIBCC memory grid above; the sampling interval is computed from these values.
const CANDIDATE_T_MIN = 0.0
const CANDIDATE_T_MAX = 5000.0
const N_CANDIDATE_SAMPLES = 2501
const CANDIDATE_ORDER = 80
const MAX_CANDIDATE_DECAY_RATE = 50.0 # fs^-1; excludes spurious stiff ESPRIT poles
const SELECTION_TOLERANCE = 1.0e-2
const MAX_SELECTED_TERMS = 30

# Direct-ESPRIT grid and fitting controls. Set DIRECT_ESPRIT_ORDER to an Int
# for a fixed-order fit, or leave it as `nothing` to use the tolerance mode.
const DIRECT_T_MIN = 0.0
const DIRECT_T_MAX = 4000.0
const N_DIRECT_SAMPLES = 2001
const DIRECT_ESPRIT_TOLERANCE = 1.0e-2
const DIRECT_ESPRIT_ORDER = nothing # e.g. 80 for fixed-order ESPRIT

# HEOM time-evolution parameters.
const T_END = 500.0
const DT_EVOLVE = 0.5
const HIERARCHY_DEPTH = 3
const HIERARCHY_TOLERANCE = 1.0e-5
const GENERATE_PLOTS = false

function site_projector(d::Int, site::Int)
    S = zeros(ComplexF64, d, d)
    S[site, site] = 1.0
    return S
end

function interaction_picture_operator(H::AbstractMatrix, S::AbstractMatrix, tau::Real)
    U = exp(-im * H * tau)
    return U * S * U'
end

relative_error(reference, approximation) =
    norm(approximation - reference) / norm(reference)

function evaluate_exponential_fit(times, poles, coefficients)
    result = zeros(ComplexF64, length(times))
    for (pole, coefficient) in zip(poles, coefficients)
        result .+= coefficient .* exp.(-pole .* times)
    end
    return result
end

function warn_nondecaying_poles(label, poles)
    indices = findall(pole -> real(pole) <= 0.0, poles)
    isempty(indices) && return indices
    @warn "$label produced poles with non-positive real parts; these terms grow or do not decay." count=length(indices) minimum_real_part=minimum(real, poles[indices]) indices=indices
    return indices
end

qfind_fmo62 = normpath(joinpath(
    @__DIR__, "..", "..", "..", "QFiND", "examples", "FMO62", "spectral_density.jl",
))
isfile(qfind_fmo62) || error("FMO62 spectral-density implementation not found: $qfind_fmo62")
include(qfind_fmo62)

outdir = @__DIR__
mkpath(outdir)

# First two localized sites of the three-site FMO Hamiltonian.
H = ComplexF64[
    310.0 -97.9
    -97.9 230.0
] * icm2ifs

sd = FMO62SpectralDensity()
tau = collect(range(0.0, T_C, length=N_TAU))
dtau = tau[2] - tau[1]

println("Two-site FMO62 SIBCC-HEOM example")
println("  Hamiltonian: first two localized FMO sites")
println("  Temperature: $TEMPERATURE K")
println("  tau grid: $N_TAU points on [0, $T_C] fs (dtau = $dtau fs)")
println("  output-time points: $N_OUTPUT_TIME")
println("  candidate ESPRIT window: [$CANDIDATE_T_MIN, $CANDIDATE_T_MAX] fs")
println("  candidate ESPRIT samples: $N_CANDIDATE_SAMPLES")
println("  candidate ESPRIT order: $CANDIDATE_ORDER")
println("  SIBCC tolerance: $SELECTION_TOLERANCE")
println("  direct ESPRIT window: [$DIRECT_T_MIN, $DIRECT_T_MAX] fs")
println("  direct ESPRIT samples: $N_DIRECT_SAMPLES")
println("  direct ESPRIT tolerance: $DIRECT_ESPRIT_TOLERANCE")
println("  direct ESPRIT fixed order: $DIRECT_ESPRIT_ORDER")

println("Sampling the full AR + 62-mode bath correlation function ...")
reference_bcf = fmo62_bcf_samples(sd, TEMPERATURE, tau)

println("Generating ESPRIT candidate exponentials ...")
candidate_times = collect(range(
    CANDIDATE_T_MIN,
    CANDIDATE_T_MAX;
    length=N_CANDIDATE_SAMPLES,
))
dt_candidate = candidate_times[2] - candidate_times[1]
candidate_bcf = fmo62_bcf_samples(sd, TEMPERATURE, candidate_times)
println("  candidate ESPRIT sampling interval: $dt_candidate fs")
candidate_fit = ExpFit.esprit(candidate_bcf, dt_candidate, CANDIDATE_ORDER)
raw_candidate_poles = ComplexF64.(candidate_fit.expon)
warn_nondecaying_poles("Candidate ESPRIT", raw_candidate_poles)
stiff_indices = findall(
    pole -> real(pole) > MAX_CANDIDATE_DECAY_RATE,
    raw_candidate_poles,
)
if !isempty(stiff_indices)
    @warn "Candidate ESPRIT produced poles above MAX_CANDIDATE_DECAY_RATE; these terms will be excluded." count=length(stiff_indices) maximum_real_part=maximum(real, raw_candidate_poles[stiff_indices]) indices=stiff_indices
end
candidate_poles = filter(
    pole -> 0.0 < real(pole) <= MAX_CANDIDATE_DECAY_RATE,
    raw_candidate_poles,
)
println("  Retained $(length(candidate_poles)) / $(length(raw_candidate_poles)) stable, nonstiff candidates")

println("Fitting the same BCF directly with ESPRIT ...")
direct_times = collect(range(
    DIRECT_T_MIN,
    DIRECT_T_MAX;
    length=N_DIRECT_SAMPLES,
))
dt_direct = direct_times[2] - direct_times[1]
direct_reference_bcf = fmo62_bcf_samples(sd, TEMPERATURE, direct_times)
println("  direct ESPRIT sampling interval: $dt_direct fs")
direct_fit = if isnothing(DIRECT_ESPRIT_ORDER)
    ExpFit.esprit(direct_reference_bcf, dt_direct, DIRECT_ESPRIT_TOLERANCE)
else
    ExpFit.esprit(direct_reference_bcf, dt_direct, DIRECT_ESPRIT_ORDER)
end
direct_poles = ComplexF64.(direct_fit.expon)
warn_nondecaying_poles("Direct ESPRIT", direct_poles)
raw_direct_coefficients = ComplexF64.(direct_fit.coeff)
# ExpFit treats the first sample as t=0. Rescale coefficients so that the
# resulting expansion is expressed in the physical time measured from t=0.
direct_coefficients = raw_direct_coefficients .* exp.(direct_poles .* DIRECT_T_MIN)
direct_bcf = evaluate_exponential_fit(tau, direct_poles, direct_coefficients)
direct_bcf_error = relative_error(reference_bcf, direct_bcf)
direct_fit_window_bcf = evaluate_exponential_fit(
    direct_times,
    direct_poles,
    direct_coefficients,
)
direct_fit_window_error = relative_error(direct_reference_bcf, direct_fit_window_bcf)
println("  Direct ESPRIT: $(length(direct_poles)) terms")
println("    BCF L2 error on direct-fit window: $direct_fit_window_error")
println("    BCF L2 error on SIBCC window: $direct_bcf_error")
save_expon_coeff(
    direct_poles,
    direct_coefficients,
    joinpath(outdir, "fmo62_direct_esprit_expon_coeff.txt"),
)

d = size(H, 1)
sibcc_baths = BathExp[]
direct_baths = BathExp[]
selected_results = Vector{Any}(undef, d)
summary_rows = Any[]

println("Applying site-resolved SIBCC group-OLS selection ...")
for site in 1:d
    S = site_projector(d, site)
    operators = [interaction_picture_operator(H, S, t) for t in tau]
    gram = sibcc_gram(S, operators, dtau, N_OUTPUT_TIME)
    selected = select_exponentials_group_ols(
        gram,
        reference_bcf,
        tau,
        candidate_poles;
        tolerance=SELECTION_TOLERANCE,
        max_terms=MAX_SELECTED_TERMS,
    )

    selected_results[site] = selected
    bcf_error = relative_error(reference_bcf, selected.fitted_bcf)
    push!(sibcc_baths, BathExp(selected.selected_poles, selected.coefficients, S))
    push!(direct_baths, BathExp(direct_poles, direct_coefficients, S))

    reference_vector = sibcc_correlation_vector(reference_bcf)
    direct_residual = sibcc_correlation_vector(direct_bcf) - reference_vector
    direct_sibcc_error = sqrt(
        dot(direct_residual, gram * direct_residual) /
        dot(reference_vector, gram * reference_vector),
    )
    push!(summary_rows, (
        "SIBCC",
        site,
        length(selected.selected_indices),
        selected.relative_error,
        bcf_error,
        join(selected.selected_indices, ","),
    ))
    push!(summary_rows, (
        "direct_ESPRIT",
        site,
        length(direct_poles),
        direct_sibcc_error,
        direct_bcf_error,
        "-",
    ))

    save_expon_coeff(
        selected.selected_poles,
        selected.coefficients,
        joinpath(outdir, "fmo62_sibcc_site$(site)_expon_coeff.txt"),
    )

    println("  Site $site: $(length(selected.selected_indices)) terms, ",
            "SIBCC error = $(selected.relative_error), BCF L2 error = $bcf_error")
    println("    Direct ESPRIT on this site's Gram: SIBCC error = $direct_sibcc_error")
end

open(joinpath(outdir, "fmo62_method_comparison.txt"), "w") do io
    println(io, "# method site terms sibcc_relative_error bcf_l2_error selected_candidate_indices")
    writedlm(io, summary_rows)
end

noise_sibcc = NoiseExp(sibcc_baths)
noise_direct = NoiseExp(direct_baths)
println("SIBCC HEOM noise: $(noise_sibcc.nbath) baths, $(noise_sibcc.nterms) expanded terms")
println("  Expanded terms per bath: $(noise_sibcc.nterms_bath)")
println("Direct-ESPRIT HEOM noise: $(noise_direct.nbath) baths, $(noise_direct.nterms) expanded terms")
println("  Expanded terms per bath: $(noise_direct.nterms_bath)")

system_sibcc = HEOMSystem(
    H,
    noise_sibcc,
    HIERARCHY_DEPTH;
    hierarchy=:width,
    tolerance=HIERARCHY_TOLERANCE,
    filter=true,
)
system_direct = HEOMSystem(
    H,
    noise_direct,
    HIERARCHY_DEPTH;
    hierarchy=:width,
    tolerance=HIERARCHY_TOLERANCE,
    filter=true,
)
println("SIBCC HEOM hierarchy: depth $HIERARCHY_DEPTH, $(system_sibcc.nado) ADOs")
println("Direct-ESPRIT HEOM hierarchy: depth $HIERARCHY_DEPTH, $(system_direct.nado) ADOs")

P0_sibcc = initial_ado(system_sibcc, 1)
P0_direct = initial_ado(system_direct, 1)
println("Running SIBCC HEOM dynamics from site 1 ...")
times_sibcc, populations_sibcc = evolve(
    system_sibcc,
    P0_sibcc,
    (0.0, T_END),
    DT_EVOLVE;
    parallel=true,
    savefile=joinpath(outdir, "pop_heom_fmo62_sibcc.dat"),
    save_interval=10,
)
println("Running direct-ESPRIT HEOM dynamics from site 1 ...")
times_direct, populations_direct = evolve(
    system_direct,
    P0_direct,
    (0.0, T_END),
    DT_EVOLVE;
    parallel=true,
    savefile=joinpath(outdir, "pop_heom_fmo62_direct_esprit.dat"),
    save_interval=10,
)
println("SIBCC final populations: rho11=$(populations_sibcc[1, end]), rho22=$(populations_sibcc[2, end])")
println("Direct-ESPRIT final populations: rho11=$(populations_direct[1, end]), rho22=$(populations_direct[2, end])")
population_difference = maximum(abs, populations_sibcc - populations_direct)
println("Maximum population difference: $population_difference")

open(joinpath(outdir, "fmo62_dynamics_comparison.txt"), "w") do io
    println(io, "# method n_bcf_terms expanded_heom_terms nado rho11_final rho22_final")
    writedlm(io, [
        ("SIBCC", sum(length(r.selected_poles) for r in selected_results),
         noise_sibcc.nterms, system_sibcc.nado,
         real(populations_sibcc[1, end]), real(populations_sibcc[2, end])),
        ("direct_ESPRIT", 2length(direct_poles), noise_direct.nterms,
         system_direct.nado,
         real(populations_direct[1, end]), real(populations_direct[2, end])),
    ])
    println(io, "# maximum_population_difference $population_difference")
end

if GENERATE_PLOTS
    @eval using CairoMakie

    fig_population = Figure(size=(800, 500))
    ax_population = Axis(
        fig_population[1, 1];
        xlabel="Time [fs]",
        ylabel="Population",
        title="Two-site FMO62 HEOM: SIBCC vs direct ESPRIT",
    )
    lines!(ax_population, times_sibcc, real.(populations_sibcc[1, :]);
           linewidth=2, label="rho11 SIBCC", color=:blue)
    lines!(ax_population, times_sibcc, real.(populations_sibcc[2, :]);
           linewidth=2, label="rho22 SIBCC", color=:red)
    lines!(ax_population, times_direct, real.(populations_direct[1, :]);
           linewidth=2, linestyle=:dash, label="rho11 direct ESPRIT", color=:deepskyblue3)
    lines!(ax_population, times_direct, real.(populations_direct[2, :]);
           linewidth=2, linestyle=:dash, label="rho22 direct ESPRIT", color=:orangered)
    axislegend(ax_population; position=:rt)
    save(joinpath(outdir, "population_fmo62_method_comparison.png"), fig_population)

    for site in 1:d
        selected = selected_results[site]
        fig_bcf = Figure(size=(800, 600))
        ax_bcf = Axis(
            fig_bcf[1, 1];
            xlabel="Time [fs]",
            ylabel="C(t)",
            title="Site $site FMO62 BCF: reference and SIBCC",
        )
        lines!(ax_bcf, tau, real.(reference_bcf); linewidth=3, linestyle=:dot,
               label="Re reference", color=:black)
        lines!(ax_bcf, tau, imag.(reference_bcf); linewidth=3, linestyle=:dot,
               label="Im reference", color=:gray40)
        lines!(ax_bcf, tau, real.(selected.fitted_bcf); linewidth=2,
               label="Re SIBCC", color=:royalblue)
        lines!(ax_bcf, tau, imag.(selected.fitted_bcf); linewidth=2,
               label="Im SIBCC", color=:orangered)
        lines!(ax_bcf, tau, real.(direct_bcf); linewidth=1.5, linestyle=:dash,
               label="Re direct ESPRIT", color=:deepskyblue3)
        lines!(ax_bcf, tau, imag.(direct_bcf); linewidth=1.5, linestyle=:dash,
               label="Im direct ESPRIT", color=:darkorange2)
        axislegend(ax_bcf; position=:rt)
        save(joinpath(outdir, "bcf_fmo62_sibcc_site$(site).png"), fig_bcf)
    end
end

println("Saved all results under $outdir")
