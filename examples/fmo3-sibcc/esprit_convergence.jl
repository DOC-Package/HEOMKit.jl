using KaisouEOM
using KaisouEOM: icm2ifs
using QFiND
using DelimitedFiles
using ExpFit
using LinearAlgebra
using Printf

const TEMPERATURE = 300.0
const T_C = 500.0
const N_TIME = 251
const ESPRIT_TOLERANCES = [5.0e-2, 2.0e-2, 1.0e-2, 5.0e-3]

const T_END = 500.0
const DT_EVOLVE = 0.5
const HIERARCHY_DEPTH = 3
const HIERARCHY_TOLERANCE = 1.0e-5

function site_projector(d::Int, site::Int)
    operator = zeros(ComplexF64, d, d)
    operator[site, site] = 1.0
    return operator
end

relative_error(reference, approximation) =
    norm(approximation - reference) / norm(reference)

function warn_nondecaying_poles(label, poles)
    indices = findall(pole -> real(pole) <= 0.0, poles)
    isempty(indices) && return indices
    @warn "$label produced poles with non-positive real parts; convergence is only assessed on the finite fitting window." count=length(indices) minimum_real_part=minimum(real, poles[indices]) indices=indices
    return indices
end

tolerance_tag(tolerance) = @sprintf("%.0e", tolerance)

qfind_fmo62 = normpath(joinpath(
    @__DIR__, "..", "..", "..", "QFiND", "examples", "FMO62", "spectral_density.jl",
))
isfile(qfind_fmo62) || error("FMO62 spectral-density implementation not found: $qfind_fmo62")
include(qfind_fmo62)

outdir = @__DIR__
H = ComplexF64[
    310.0 -97.9   5.5
    -97.9 230.0  30.1
      5.5  30.1   0.0
] * icm2ifs

sample_times = collect(range(0.0, T_C, length=N_TIME))
dt_sample = sample_times[2] - sample_times[1]
sd = FMO62SpectralDensity()

println("Sampling the full FMO62 BCF on [0, $T_C] fs ...")
reference_bcf = fmo62_bcf_samples(sd, TEMPERATURE, sample_times)

results = Any[]
d = size(H, 1)

for tolerance in ESPRIT_TOLERANCES
    tag = tolerance_tag(tolerance)
    println()
    println("Direct ESPRIT tolerance = $tolerance")

    fit = ExpFit.esprit(reference_bcf, dt_sample, tolerance)
    exponents = ComplexF64.(fit.expon)
    nondecaying_indices = warn_nondecaying_poles(
        "Direct ESPRIT (tolerance=$tolerance)",
        exponents,
    )
    coefficients = ComplexF64.(fit.coeff)
    fitted_bcf = ComplexF64.(fit.(sample_times))
    bcf_l2_error = relative_error(reference_bcf, fitted_bcf)
    bcf_max_error = maximum(abs, fitted_bcf - reference_bcf) / abs(reference_bcf[1])
    unstable_poles = length(nondecaying_indices)

    save_expon_coeff(
        exponents,
        coefficients,
        joinpath(outdir, "fmo62_direct_esprit_tol_$(tag)_expon_coeff.txt"),
    )

    baths = [
        BathExp(exponents, coefficients, site_projector(d, site))
        for site in 1:d
    ]
    noise = NoiseExp(baths)
    system = HEOMSystem(
        H,
        noise,
        HIERARCHY_DEPTH;
        hierarchy=:width,
        tolerance=HIERARCHY_TOLERANCE,
        filter=true,
    )

    println("  terms per bath: $(length(exponents))")
    println("  BCF L2 error: $bcf_l2_error")
    println("  nondecaying poles: $unstable_poles")
    println("  expanded HEOM terms: $(noise.nterms)")
    println("  ADOs: $(system.nado)")

    initial_state = initial_ado(system, 1)
    times, populations = evolve(
        system,
        initial_state,
        (0.0, T_END),
        DT_EVOLVE;
        parallel=true,
        savefile=joinpath(outdir, "pop_heom_fmo62_direct_esprit_tol_$(tag).dat"),
        save_interval=10,
    )

    push!(results, (
        tolerance=tolerance,
        tag=tag,
        nterms=length(exponents),
        bcf_l2_error=bcf_l2_error,
        bcf_max_error=bcf_max_error,
        unstable_poles=unstable_poles,
        expanded_terms=noise.nterms,
        nado=system.nado,
        times=times,
        populations=populations,
    ))
    println("  final populations: rho11=$(populations[1, end]), rho22=$(populations[2, end]), rho33=$(populations[3, end])")

    system = nothing
    noise = nothing
    GC.gc()
end

reference_populations = results[end].populations
summary_rows = Any[]
for result in results
    population_difference = maximum(abs, result.populations - reference_populations)
    push!(summary_rows, (
        result.tolerance,
        result.nterms,
        result.bcf_l2_error,
        result.bcf_max_error,
        result.unstable_poles,
        result.expanded_terms,
        result.nado,
        real(result.populations[1, end]),
        real(result.populations[2, end]),
        real(result.populations[3, end]),
        population_difference,
    ))
end

summary_file = joinpath(outdir, "fmo62_direct_esprit_convergence.txt")
open(summary_file, "w") do io
    println(io, "# tolerance terms_per_bath bcf_l2_error bcf_max_error nondecaying_poles expanded_heom_terms nado rho11_final rho22_final rho33_final max_population_difference_vs_tol_5e-03")
    writedlm(io, summary_rows)
end

println()
println("Saved convergence summary: $summary_file")
