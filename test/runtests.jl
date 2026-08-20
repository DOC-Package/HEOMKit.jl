using KaisouEOM
using Test

@testset "KaisouEOM.jl" begin
    include("blas.jl")
    include("noise.jl")
    include("hierarchy.jl")
end
