using KaisouEOM
using Test

@testset "hierarchy.jl" begin
    
    # Test the hierarchy_index_depth function
    nbranch = 2
    ndepth = 2
    nado, nvec, npvec, nmvec = hierarchy_index_depth(nbranch, ndepth)

    # print the results
    println("nado: ", nado)
    println("nvec: ", transpose(nvec))
    println("npvec: ", transpose(npvec))
    println("nmvec: ", transpose(nmvec))
end
