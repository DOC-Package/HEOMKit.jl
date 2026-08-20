using CairoMakie
using DelimitedFiles

function read_numeric_table(filename)
    return Float64.(readdlm(filename; comments=true, comment_char='#'))
end

outdir = @__DIR__
population_sibcc = read_numeric_table(joinpath(outdir, "pop_ohmic_sibcc.dat"))
population_direct = read_numeric_table(joinpath(outdir, "pop_ohmic_direct_esprit.dat"))
bcf = read_numeric_table(joinpath(outdir, "ohmic_bcf_comparison.dat"))

figure_population = Figure(size=(900, 720))
axis_population = Axis(
    figure_population[1, 1];
    xlabel="",
    ylabel="Population",
    title="Ohmic exponential-cutoff HEOM: SIBCC vs direct ESPRIT",
)
axis_difference = Axis(
    figure_population[2, 1];
    xlabel="Time [fs]",
    ylabel="rho11(SIBCC) - rho11(direct)",
)
lines!(axis_population, population_sibcc[:, 1], population_sibcc[:, 2];
       linewidth=3, color=:royalblue, label="rho11 SIBCC")
lines!(axis_population, population_sibcc[:, 1], population_sibcc[:, 3];
       linewidth=3, color=:orangered, label="rho22 SIBCC")
lines!(axis_population, population_direct[:, 1], population_direct[:, 2];
       linewidth=2.5, linestyle=:dash, color=:deepskyblue3, label="rho11 direct ESPRIT")
lines!(axis_population, population_direct[:, 1], population_direct[:, 3];
       linewidth=2.5, linestyle=:dash, color=:darkorange2, label="rho22 direct ESPRIT")
lines!(axis_difference, population_sibcc[:, 1],
       population_sibcc[:, 2] .- population_direct[:, 2];
       linewidth=2.5, color=:purple)
axislegend(axis_population; position=:rt)
save(joinpath(outdir, "population_ohmic_sibcc_comparison.png"), figure_population)

normalization = hypot(bcf[1, 2], bcf[1, 3])
figure_bcf = Figure(size=(900, 850))
axis_real = Axis(figure_bcf[1, 1]; xlabel="", ylabel="Re C(t) / |C(0)|",
                 title="Ohmic BCF: SIBCC vs direct ESPRIT")
axis_imag = Axis(figure_bcf[2, 1]; xlabel="", ylabel="Im C(t) / |C(0)|")
axis_error = Axis(figure_bcf[3, 1]; xlabel="Time [fs]",
                  ylabel="|delta C(t)| / |C(0)|", yscale=log10)

lines!(axis_real, bcf[:, 1], bcf[:, 2] ./ normalization;
       linewidth=3, color=:black, label="Reference")
lines!(axis_real, bcf[:, 1], bcf[:, 4] ./ normalization;
       linewidth=2.5, color=:royalblue, label="SIBCC")
lines!(axis_real, bcf[:, 1], bcf[:, 6] ./ normalization;
       linewidth=2.5, linestyle=:dash, color=:orangered, label="Direct ESPRIT")
lines!(axis_imag, bcf[:, 1], bcf[:, 3] ./ normalization;
       linewidth=3, color=:black)
lines!(axis_imag, bcf[:, 1], bcf[:, 5] ./ normalization;
       linewidth=2.5, color=:royalblue)
lines!(axis_imag, bcf[:, 1], bcf[:, 7] ./ normalization;
       linewidth=2.5, linestyle=:dash, color=:orangered)

sibcc_error = hypot.(bcf[:, 4] .- bcf[:, 2], bcf[:, 5] .- bcf[:, 3]) ./ normalization
direct_error = hypot.(bcf[:, 6] .- bcf[:, 2], bcf[:, 7] .- bcf[:, 3]) ./ normalization
lines!(axis_error, bcf[:, 1], max.(sibcc_error, eps(Float64));
       linewidth=2.5, color=:royalblue, label="SIBCC")
lines!(axis_error, bcf[:, 1], max.(direct_error, eps(Float64));
       linewidth=2.5, linestyle=:dash, color=:orangered, label="Direct ESPRIT")
axislegend(axis_real; position=:rt)
save(joinpath(outdir, "bcf_ohmic_sibcc_comparison.png"), figure_bcf)

println("Saved Ohmic SIBCC validation plots.")

