using HEOMKit
using Test

@testset "HEOMKit.jl" begin
    include("blas.jl")
    include("noise.jl")
    include("hierarchy.jl")
end
