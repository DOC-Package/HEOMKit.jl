using HEOMKit
using HEOMKit: icm2ifs
using QFiND
using DelimitedFiles
using ExpFit
using LinearAlgebra

# Ohmic spectral density J(w) with an exponential cutoff.
const TEMPERATURE = 300.0
const CUTOFF_FREQUENCY = 50.0 # cm^-1
const COUPLING_STRENGTH = 10.0 # QFiND PowerLawExpSD parameter, cm^-1
const BCF_UPPER_FREQUENCY = 1000.0 # cm^-1

# Correlation/SIBCC grid.
const T_C = 1000.0 # fs
const N_TAU = 501
const N_OUTPUT_TIME = 501
const CANDIDATE_ORDER = 20
const SIBCC_TOLERANCE = 7.0e-2
const MAX_SELECTED_TERMS = 10
const DIRECT_ESPRIT_TOLERANCE = 5.0e-1

# HEOM propagation.
const T_END = 1000.0 # fs
const DT_EVOLVE = 0.5 # fs
const HIERARCHY_DEPTH = 5
const HIERARCHY_TOLERANCE = 1.0e-8

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

outdir = @__DIR__
mkpath(outdir)

# Unbiased two-level system H = -(Delta/2) sigma_x, Delta = 20 cm^-1.
H = ComplexF64[
     0.0 -10.0
    -10.0   0.0
] * icm2ifs
S = ComplexF64[
    1.0  0.0
    0.0 -1.0
]

sd = PowerLawExpSD(1.0, CUTOFF_FREQUENCY; reorgene=COUPLING_STRENGTH)
bcf = BosonicBCF(
    sd,
    TEMPERATURE;
    ub=BCF_UPPER_FREQUENCY,
    rtol=1.0e-8,
)

tau = collect(range(0.0, T_C, length=N_TAU))
dtau = tau[2] - tau[1]

println("Ohmic-exponential-cutoff SIBCC validation")
println("  cutoff frequency: $CUTOFF_FREQUENCY cm^-1")
println("  coupling parameter: $COUPLING_STRENGTH cm^-1")
println("  temperature: $TEMPERATURE K")
println("  tau grid: $N_TAU points on [0, $T_C] fs (dtau=$dtau fs)")

println("Sampling the reference BCF ...")
reference_bcf = ComplexF64[bcf(t) for t in tau]

println("Generating ESPRIT candidates ...")
candidate_fit = ExpFit.esprit(reference_bcf, dtau, CANDIDATE_ORDER)
raw_candidate_poles = ComplexF64.(candidate_fit.expon)
nonpositive_candidates = findall(pole -> real(pole) <= 0.0, raw_candidate_poles)
if !isempty(nonpositive_candidates)
    @warn "Candidate ESPRIT produced nondecaying poles; they will be excluded." count=length(nonpositive_candidates) indices=nonpositive_candidates minimum_real_part=minimum(real, raw_candidate_poles[nonpositive_candidates])
end
candidate_poles = filter(pole -> real(pole) > 0.0, raw_candidate_poles)
println("  stable candidates: $(length(candidate_poles)) / $(length(raw_candidate_poles))")

operators = [interaction_picture_operator(H, S, t) for t in tau]
gram = sibcc_gram(S, operators, dtau, N_OUTPUT_TIME)
selected = select_exponentials_group_ols(
    gram,
    reference_bcf,
    tau,
    candidate_poles;
    tolerance=SIBCC_TOLERANCE,
    max_terms=MAX_SELECTED_TERMS,
)
sibcc_bcf = selected.fitted_bcf
sibcc_bcf_error = relative_error(reference_bcf, sibcc_bcf)

println("SIBCC result")
println("  selected terms: $(length(selected.selected_poles))")
println("  selected candidate indices: $(selected.selected_indices)")
println("  SIBCC relative error: $(selected.relative_error)")
println("  Euclidean BCF error: $sibcc_bcf_error")

println("Fitting the BCF directly with ESPRIT ...")
direct_fit = ExpFit.esprit(reference_bcf, dtau, DIRECT_ESPRIT_TOLERANCE)
direct_poles = ComplexF64.(direct_fit.expon)
direct_coefficients = ComplexF64.(direct_fit.coeff)
nonpositive_direct = findall(pole -> real(pole) <= 0.0, direct_poles)
if !isempty(nonpositive_direct)
    @warn "Direct ESPRIT produced nondecaying poles." count=length(nonpositive_direct) indices=nonpositive_direct minimum_real_part=minimum(real, direct_poles[nonpositive_direct])
end
direct_bcf = evaluate_exponential_fit(tau, direct_poles, direct_coefficients)
direct_bcf_error = relative_error(reference_bcf, direct_bcf)

reference_vector = sibcc_correlation_vector(reference_bcf)
direct_residual = sibcc_correlation_vector(direct_bcf) - reference_vector
direct_sibcc_error = sqrt(
    dot(direct_residual, gram * direct_residual) /
    dot(reference_vector, gram * reference_vector),
)

println("Direct ESPRIT result")
println("  terms: $(length(direct_poles))")
println("  SIBCC-weighted error: $direct_sibcc_error")
println("  Euclidean BCF error: $direct_bcf_error")

save_expon_coeff(
    selected.selected_poles,
    selected.coefficients,
    joinpath(outdir, "ohmic_sibcc_expon_coeff.txt"),
)
save_expon_coeff(
    direct_poles,
    direct_coefficients,
    joinpath(outdir, "ohmic_direct_esprit_expon_coeff.txt"),
)

open(joinpath(outdir, "ohmic_bcf_comparison.dat"), "w") do io
    println(io, "# time ref_real ref_imag sibcc_real sibcc_imag direct_real direct_imag")
    writedlm(io, hcat(
        tau,
        real.(reference_bcf), imag.(reference_bcf),
        real.(sibcc_bcf), imag.(sibcc_bcf),
        real.(direct_bcf), imag.(direct_bcf),
    ))
end

noise_sibcc = NoiseExp(BathExp(selected.selected_poles, selected.coefficients, S))
noise_direct = NoiseExp(BathExp(direct_poles, direct_coefficients, S))

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
    savefile=joinpath(outdir, "pop_ohmic_sibcc.dat"),
    save_interval=10,
)

println("Running direct-ESPRIT HEOM dynamics ...")
times_direct, populations_direct = evolve(
    system_direct,
    initial_ado(system_direct, 1),
    (0.0, T_END),
    DT_EVOLVE;
    parallel=true,
    savefile=joinpath(outdir, "pop_ohmic_direct_esprit.dat"),
    save_interval=10,
)

maximum_population_difference = maximum(
    abs,
    populations_sibcc - populations_direct,
)

summary_file = joinpath(outdir, "ohmic_sibcc_summary.txt")
open(summary_file, "w") do io
    println(io, "# method terms expanded_heom_terms nado system_weighted_error bcf_l2_error rho11_final rho22_final")
    writedlm(io, [
        ("SIBCC", length(selected.selected_poles), noise_sibcc.nterms,
         system_sibcc.nado, selected.relative_error, sibcc_bcf_error,
         real(populations_sibcc[1, end]), real(populations_sibcc[2, end])),
        ("direct_ESPRIT", length(direct_poles), noise_direct.nterms,
         system_direct.nado, direct_sibcc_error, direct_bcf_error,
         real(populations_direct[1, end]), real(populations_direct[2, end])),
    ])
    println(io, "# maximum_population_difference $maximum_population_difference")
end

println("Final populations")
println("  SIBCC: rho11=$(populations_sibcc[1, end]), rho22=$(populations_sibcc[2, end])")
println("  Direct: rho11=$(populations_direct[1, end]), rho22=$(populations_direct[2, end])")
println("  Maximum population difference: $maximum_population_difference")
println("Saved summary: $summary_file")

