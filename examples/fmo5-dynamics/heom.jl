using HEOMKit
using HEOMKit: icm2ifs
using QFiND
using DelimitedFiles
using ExpFit
using LinearAlgebra

const TEMPERATURE = 300.0

# The same 500 fs BCF samples are used for candidate ESPRIT, SIBCC, and the
# direct-ESPRIT reference calculation.
const T_C = 500.0
const N_TAU = 251
const N_OUTPUT_TIME = 251
const CANDIDATE_ORDER = 10
const SIBCC_TOLERANCE = 2.0e-2
const MAX_SELECTED_TERMS = 7
const DIRECT_ESPRIT_TOLERANCE = 1.0e-2

const T_END = 500.0
const DT_EVOLVE = 0.5
const HIERARCHY_DEPTH = 3
const HIERARCHY_TOLERANCE = 1.0e-5

function site_projector(d::Int, site::Int)
    operator = zeros(ComplexF64, d, d)
    operator[site, site] = 1.0
    return operator
end

function interaction_picture_operator(H, S, time)
    U = exp(-im * H * time)
    return U * S * U'
end

function evaluate_exponential_fit(times, poles, coefficients)
    result = zeros(ComplexF64, length(times))
    for (pole, coefficient) in zip(poles, coefficients)
        result .+= coefficient .* exp.(-pole .* times)
    end
    return result
end

relative_error(reference, approximation) =
    norm(approximation - reference) / norm(reference)

qfind_fmo5 = normpath(joinpath(
    @__DIR__, "..", "..", "..", "QFiND", "examples", "FMO5", "spectral_density.jl",
))
isfile(qfind_fmo5) || error("FMO5 spectral-density implementation not found: $qfind_fmo5")
include(qfind_fmo5)

outdir = @__DIR__
H = ComplexF64[
    310.0 -97.9
    -97.9 230.0
] * icm2ifs
d = size(H, 1)

sd = FMO5SpectralDensity()
tau = collect(range(0.0, T_C, length=N_TAU))
dtau = tau[2] - tau[1]

println("Two-site FMO5 dynamics: SIBCC vs direct ESPRIT")
println("  Temperature: $TEMPERATURE K")
println("  Common BCF grid: $N_TAU points on [0, $T_C] fs (dt=$dtau fs)")

println("Sampling the FMO5 BCF ...")
reference_bcf = fmo5_bcf_samples(sd, TEMPERATURE, tau)

println("Generating fixed-order ESPRIT candidates ...")
candidate_fit = ExpFit.esprit(reference_bcf, dtau, CANDIDATE_ORDER)
raw_candidate_poles = ComplexF64.(candidate_fit.expon)
nonpositive_candidates = findall(pole -> real(pole) <= 0.0, raw_candidate_poles)
if !isempty(nonpositive_candidates)
    @warn "Candidate ESPRIT produced non-positive-real poles; excluding them." count=length(nonpositive_candidates) indices=nonpositive_candidates minimum_real_part=minimum(real, raw_candidate_poles[nonpositive_candidates])
end
candidate_poles = filter(pole -> real(pole) > 0.0, raw_candidate_poles)
println("  Stable candidates: $(length(candidate_poles)) / $(length(raw_candidate_poles))")

println("Applying site-resolved SIBCC selection ...")
sibcc_baths = BathExp[]
selected_results = Vector{Any}(undef, d)
gram_by_site = Vector{Matrix{Float64}}(undef, d)

for site in 1:d
    operator = site_projector(d, site)
    interaction_operators = [
        interaction_picture_operator(H, operator, time)
        for time in tau
    ]
    gram = sibcc_gram(operator, interaction_operators, dtau, N_OUTPUT_TIME)
    gram_by_site[site] = gram
    selected = select_exponentials_group_ols(
        gram,
        reference_bcf,
        tau,
        candidate_poles;
        tolerance=SIBCC_TOLERANCE,
        max_terms=MAX_SELECTED_TERMS,
    )
    selected_results[site] = selected
    push!(sibcc_baths, BathExp(
        selected.selected_poles,
        selected.coefficients,
        operator,
    ))

    bcf_error = relative_error(reference_bcf, selected.fitted_bcf)
    println("  Site $site: $(length(selected.selected_poles)) terms, ",
            "SIBCC error=$(selected.relative_error), BCF error=$bcf_error")
    save_expon_coeff(
        selected.selected_poles,
        selected.coefficients,
        joinpath(outdir, "fmo5_sibcc_site$(site)_expon_coeff.txt"),
    )
end

println("Applying direct ESPRIT on the same 500 fs samples ...")
direct_fit = ExpFit.esprit(reference_bcf, dtau, DIRECT_ESPRIT_TOLERANCE)
direct_poles = ComplexF64.(direct_fit.expon)
direct_coefficients = ComplexF64.(direct_fit.coeff)
nonpositive_direct = findall(pole -> real(pole) <= 0.0, direct_poles)
isempty(nonpositive_direct) || error(
    "Direct ESPRIT produced non-positive-real poles at $nonpositive_direct",
)
direct_bcf = evaluate_exponential_fit(tau, direct_poles, direct_coefficients)
direct_bcf_error = relative_error(reference_bcf, direct_bcf)
println("  Direct ESPRIT: $(length(direct_poles)) terms, BCF error=$direct_bcf_error")
save_expon_coeff(
    direct_poles,
    direct_coefficients,
    joinpath(outdir, "fmo5_direct_esprit_expon_coeff.txt"),
)

direct_baths = [
    BathExp(direct_poles, direct_coefficients, site_projector(d, site))
    for site in 1:d
]

method_rows = Any[]
for site in 1:d
    selected = selected_results[site]
    reference_vector = sibcc_correlation_vector(reference_bcf)
    direct_residual = sibcc_correlation_vector(direct_bcf) - reference_vector
    direct_sibcc_error = sqrt(
        dot(direct_residual, gram_by_site[site] * direct_residual) /
        dot(reference_vector, gram_by_site[site] * reference_vector),
    )
    push!(method_rows, (
        "SIBCC", site, length(selected.selected_poles),
        selected.relative_error,
        relative_error(reference_bcf, selected.fitted_bcf),
    ))
    push!(method_rows, (
        "direct_ESPRIT", site, length(direct_poles),
        direct_sibcc_error, direct_bcf_error,
    ))

    open(joinpath(outdir, "fmo5_bcf_site$(site).dat"), "w") do io
        println(io, "# time ref_real ref_imag sibcc_real sibcc_imag direct_real direct_imag")
        writedlm(io, hcat(
            tau,
            real.(reference_bcf), imag.(reference_bcf),
            real.(selected.fitted_bcf), imag.(selected.fitted_bcf),
            real.(direct_bcf), imag.(direct_bcf),
        ))
    end
end

open(joinpath(outdir, "fmo5_method_comparison.txt"), "w") do io
    println(io, "# method site terms sibcc_relative_error bcf_l2_error")
    writedlm(io, method_rows)
end

noise_sibcc = NoiseExp(sibcc_baths)
noise_direct = NoiseExp(direct_baths)
system_sibcc = HEOMSystem(
    H, noise_sibcc, HIERARCHY_DEPTH;
    hierarchy=:width, tolerance=HIERARCHY_TOLERANCE, filter=true,
)
system_direct = HEOMSystem(
    H, noise_direct, HIERARCHY_DEPTH;
    hierarchy=:width, tolerance=HIERARCHY_TOLERANCE, filter=true,
)

println("HEOM hierarchy")
println("  SIBCC: $(noise_sibcc.nterms) expanded terms, $(system_sibcc.nado) ADOs")
println("  Direct: $(noise_direct.nterms) expanded terms, $(system_direct.nado) ADOs")

println("Running SIBCC HEOM dynamics ...")
times_sibcc, populations_sibcc = evolve(
    system_sibcc,
    initial_ado(system_sibcc, 1),
    (0.0, T_END),
    DT_EVOLVE;
    parallel=true,
    savefile=joinpath(outdir, "pop_heom_fmo5_sibcc.dat"),
    save_interval=10,
)

println("Running direct-ESPRIT HEOM dynamics ...")
times_direct, populations_direct = evolve(
    system_direct,
    initial_ado(system_direct, 1),
    (0.0, T_END),
    DT_EVOLVE;
    parallel=true,
    savefile=joinpath(outdir, "pop_heom_fmo5_direct_esprit.dat"),
    save_interval=10,
)

maximum_population_difference = maximum(
    abs,
    populations_sibcc - populations_direct,
)

summary_file = joinpath(outdir, "fmo5_dynamics_comparison.txt")
open(summary_file, "w") do io
    println(io, "# method terms_total expanded_heom_terms nado rho11_final rho22_final")
    writedlm(io, [
        ("SIBCC", sum(length(result.selected_poles) for result in selected_results),
         noise_sibcc.nterms, system_sibcc.nado,
         real(populations_sibcc[1, end]), real(populations_sibcc[2, end])),
        ("direct_ESPRIT", d * length(direct_poles), noise_direct.nterms,
         system_direct.nado,
         real(populations_direct[1, end]), real(populations_direct[2, end])),
    ])
    println(io, "# maximum_population_difference $maximum_population_difference")
end

println("Final populations")
println("  SIBCC: rho11=$(populations_sibcc[1, end]), rho22=$(populations_sibcc[2, end])")
println("  Direct: rho11=$(populations_direct[1, end]), rho22=$(populations_direct[2, end])")
println("  Maximum population difference: $maximum_population_difference")
println("Saved: $summary_file")

