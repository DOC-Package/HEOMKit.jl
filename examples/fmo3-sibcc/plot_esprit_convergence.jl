using CairoMakie
using DelimitedFiles
using Printf
using QFiND

const TEMPERATURE = 300.0
const T_C = 500.0
const DTAU = 2.0
const ESPRIT_TOLERANCES = [5.0e-2, 2.0e-2, 1.0e-2, 5.0e-3]

tolerance_tag(tolerance) = @sprintf("%.0e", tolerance)

function read_numeric_table(filename)
    return Float64.(readdlm(filename; comments=true, comment_char='#'))
end

function load_exponential_fit(filename)
    table = read_numeric_table(filename)
    coefficients = ComplexF64.(table[:, 1] .+ im .* table[:, 2])
    exponents = ComplexF64.(table[:, 3] .+ im .* table[:, 4])
    return exponents, coefficients
end

function evaluate_fit(times, exponents, coefficients)
    result = zeros(ComplexF64, length(times))
    for (exponent, coefficient) in zip(exponents, coefficients)
        result .+= coefficient .* exp.(-exponent .* times)
    end
    return result
end

qfind_fmo62 = normpath(joinpath(
    @__DIR__, "..", "..", "..", "QFiND", "examples", "FMO62", "spectral_density.jl",
))
include(qfind_fmo62)

outdir = @__DIR__
population_data = [
    read_numeric_table(joinpath(
        outdir, "pop_heom_fmo62_direct_esprit_tol_$(tolerance_tag(tolerance)).dat",
    ))
    for tolerance in ESPRIT_TOLERANCES
]
reference_population = population_data[end]
colors = [:darkorange, :seagreen3, :royalblue, :purple]

fig_population = Figure(size=(900, 760))
ax_population = Axis(
    fig_population[1, 1];
    xlabel="",
    ylabel="rho11(t)",
    title="Direct ESPRIT HEOM convergence for the three-site FMO62 model",
)
ax_difference = Axis(
    fig_population[2, 1];
    xlabel="Time [fs]",
    ylabel="rho11(t) - rho11(tol=5e-3)",
)
for (index, tolerance) in pairs(ESPRIT_TOLERANCES)
    data = population_data[index]
    label = "tol=$(tolerance_tag(tolerance))"
    lines!(ax_population, data[:, 1], data[:, 2];
           linewidth=2.5, color=colors[index], label=label)
    lines!(ax_difference, data[:, 1], data[:, 2] .- reference_population[:, 2];
           linewidth=2.5, color=colors[index])
end
axislegend(ax_population; position=:rt)
save(joinpath(outdir, "population_fmo62_direct_esprit_convergence.png"), fig_population)

times = collect(0.0:DTAU:T_C)
sd = FMO62SpectralDensity()
reference_bcf = fmo62_bcf_samples(sd, TEMPERATURE, times)
normalization = abs(reference_bcf[1])

fig_bcf = Figure(size=(900, 760))
ax_bcf = Axis(
    fig_bcf[1, 1];
    xlabel="",
    ylabel="Re C(t) / |C(0)|",
    title="Direct ESPRIT BCF convergence for FMO62",
)
ax_bcf_error = Axis(
    fig_bcf[2, 1];
    xlabel="Time [fs]",
    ylabel="|delta C(t)| / |C(0)|",
    yscale=log10,
)
lines!(ax_bcf, times, real.(reference_bcf) ./ normalization;
       linewidth=3, color=:black, label="Reference")
for (index, tolerance) in pairs(ESPRIT_TOLERANCES)
    tag = tolerance_tag(tolerance)
    exponents, coefficients = load_exponential_fit(
        joinpath(outdir, "fmo62_direct_esprit_tol_$(tag)_expon_coeff.txt"),
    )
    fitted_bcf = evaluate_fit(times, exponents, coefficients)
    lines!(ax_bcf, times, real.(fitted_bcf) ./ normalization;
           linewidth=2, color=colors[index], label="tol=$tag")
    error = max.(abs.(fitted_bcf - reference_bcf) ./ normalization, eps(Float64))
    lines!(ax_bcf_error, times, error; linewidth=2, color=colors[index])
end
axislegend(ax_bcf; position=:rt)
save(joinpath(outdir, "bcf_fmo62_direct_esprit_convergence.png"), fig_bcf)

println("Saved direct-ESPRIT convergence plots.")
