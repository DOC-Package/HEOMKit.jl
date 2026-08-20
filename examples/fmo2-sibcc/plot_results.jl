using CairoMakie
using DelimitedFiles
using LinearAlgebra
using QFiND

const TEMPERATURE = 300.0
const T_C = 500.0
const DTAU = 2.0

function load_exponential_fit(filename)
    table = Float64.(readdlm(filename; comments=true, comment_char='#'))
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
population_sibcc = Float64.(readdlm(
    joinpath(outdir, "pop_heom_fmo62_sibcc.dat"); comments=true, comment_char='#',
))
population_direct = Float64.(readdlm(
    joinpath(outdir, "pop_heom_fmo62_direct_esprit1.dat"); comments=true, comment_char='#',
))

fig_population = Figure(size=(900, 560))
ax_population = Axis(
    fig_population[1, 1];
    xlabel="Time [fs]",
    ylabel="Population",
    title="Two-site FMO62 HEOM: SIBCC vs direct ESPRIT",
)
lines!(ax_population, population_sibcc[:, 1], population_sibcc[:, 2];
       linewidth=3, label="rho11 SIBCC", color=:royalblue)
lines!(ax_population, population_sibcc[:, 1], population_sibcc[:, 3];
       linewidth=3, label="rho22 SIBCC", color=:orangered)
lines!(ax_population, population_direct[:, 1], population_direct[:, 2];
       linewidth=2.5, linestyle=:dash, label="rho11 direct ESPRIT", color=:deepskyblue3)
lines!(ax_population, population_direct[:, 1], population_direct[:, 3];
       linewidth=2.5, linestyle=:dash, label="rho22 direct ESPRIT", color=:darkorange2)
axislegend(ax_population; position=:rt)
save(joinpath(outdir, "population_fmo62_method_comparison.png"), fig_population)

times = collect(0.0:DTAU:T_C)
sd = FMO62SpectralDensity()
reference_bcf = fmo62_bcf_samples(sd, TEMPERATURE, times)
direct_exponents, direct_coefficients = load_exponential_fit(
    joinpath(outdir, "fmo62_direct_esprit_expon_coeff.txt"),
)
direct_bcf = evaluate_fit(times, direct_exponents, direct_coefficients)

for site in 1:2
    sibcc_exponents, sibcc_coefficients = load_exponential_fit(
        joinpath(outdir, "fmo62_sibcc_site$(site)_expon_coeff.txt"),
    )
    sibcc_bcf = evaluate_fit(times, sibcc_exponents, sibcc_coefficients)

    normalization = abs(reference_bcf[1])
    fig_bcf = Figure(size=(900, 720))
    ax_real = Axis(
        fig_bcf[1, 1];
        xlabel="",
        ylabel="Re C(t) / |C(0)|",
        title="Site $site FMO62 BCF: SIBCC vs direct ESPRIT",
    )
    ax_imag = Axis(
        fig_bcf[2, 1];
        xlabel="Time [fs]",
        ylabel="Im C(t) / |C(0)|",
    )

    lines!(ax_real, times, real.(reference_bcf) ./ normalization;
           linewidth=3, color=:black, label="Reference")
    lines!(ax_real, times, real.(sibcc_bcf) ./ normalization;
           linewidth=2.5, color=:royalblue, label="SIBCC")
    lines!(ax_real, times, real.(direct_bcf) ./ normalization;
           linewidth=2.5, linestyle=:dash, color=:orangered, label="Direct ESPRIT")
    lines!(ax_imag, times, imag.(reference_bcf) ./ normalization;
           linewidth=3, color=:black, label="Reference")
    lines!(ax_imag, times, imag.(sibcc_bcf) ./ normalization;
           linewidth=2.5, color=:royalblue, label="SIBCC")
    lines!(ax_imag, times, imag.(direct_bcf) ./ normalization;
           linewidth=2.5, linestyle=:dash, color=:orangered, label="Direct ESPRIT")
    axislegend(ax_real; position=:rt)
    save(joinpath(outdir, "bcf_fmo62_method_comparison_site$(site).png"), fig_bcf)
end

println("Saved:")
println("  ", joinpath(outdir, "population_fmo62_method_comparison.png"))
println("  ", joinpath(outdir, "bcf_fmo62_method_comparison_site1.png"))
println("  ", joinpath(outdir, "bcf_fmo62_method_comparison_site2.png"))
