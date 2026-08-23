using HEOMKit
using Test

@testset "hierarchy.jl" begin
    
    @testset "hierarchy_index_depth" begin
        # Test the hierarchy_index_depth function
        nmode = 2
        ndepth = 2
        nado, ado_idx, idx_plus, idx_minus = hierarchy_index_depth(nmode, ndepth)

        # print the results
        println("=== hierarchy_index_depth ===")
        println("nado: ", nado)
        println("ado_idx: ", transpose(ado_idx))
        println("idx_plus: ", transpose(idx_plus))
        println("idx_minus: ", transpose(idx_minus))

        # Basic tests
        @test nado == binomial(ndepth + nmode, nmode)
        @test size(ado_idx) == (nmode, nado)
        @test ado_idx[:, 1] == [0, 0]  # First index should be zero vector
    end

    @testset "hierarchy_index_width" begin
        # Test the hierarchy_index_width function
        nmode = 2
        ndepth = 2
        
        # Mock parameters for testing
        gamk = [1.0, 2.0]
        ck = [0.5, 0.5]
        bk = [1.0, 1.0]
        
        # Without filtering (should match depth-first result)
        nado_w, ado_idx_w, idx_plus_w, idx_minus_w = hierarchy_index_width(
            nmode, ndepth, gamk, ck, bk; filter=false
        )

        println("\n=== hierarchy_index_width (no filter) ===")
        println("nado: ", nado_w)
        println("ado_idx: ", transpose(ado_idx_w))
        println("idx_plus: ", transpose(idx_plus_w))
        println("idx_minus: ", transpose(idx_minus_w))

        # Basic tests
        @test nado_w == binomial(ndepth + nmode, nmode)
        @test size(ado_idx_w) == (nmode, nado_w)
        @test ado_idx_w[:, 1] == [0, 0]  # First index should be zero vector

        # Test connectivity: idx_plus and idx_minus should be consistent
        for n in 1:nado_w
            for k in 1:nmode
                np = idx_plus_w[k, n]
                if np > 0
                    # Forward connection: ado_idx_w[:, np] should be ado_idx_w[:, n] with k-th element +1
                    expected = copy(ado_idx_w[:, n])
                    expected[k] += 1
                    @test ado_idx_w[:, np] == expected
                end
                
                nm = idx_minus_w[k, n]
                if nm > 0
                    # Backward connection: ado_idx_w[:, nm] should be ado_idx_w[:, n] with k-th element -1
                    expected = copy(ado_idx_w[:, n])
                    expected[k] -= 1
                    @test ado_idx_w[:, nm] == expected
                end
            end
        end

        # Test with filtering enabled
        nado_f, ado_idx_f, idx_plus_f, idx_minus_f = hierarchy_index_width(
            nmode, ndepth, gamk, ck, bk;
            ak=1.0, tolerance=1e-6, filter=true
        )

        println("\n=== hierarchy_index_width (with filter) ===")
        println("nado: ", nado_f)
        println("ado_idx: ", transpose(ado_idx_f))

        @test nado_f > 0
        @test ado_idx_f[:, 1] == [0, 0]  # First index should always be zero vector
    end

    @testset "hierarchy_index_width vs depth comparison" begin
        # Compare width and depth methods (without filtering, they should produce same set of indices)
        nmode = 3
        ndepth = 3
        gamk = ones(nmode)
        ck = ones(nmode)
        bk = ones(nmode)

        nado_d, ado_idx_d, _, _ = hierarchy_index_depth(nmode, ndepth)
        nado_w, ado_idx_w, _, _ = hierarchy_index_width(nmode, ndepth, gamk, ck, bk; filter=false)

        @test nado_d == nado_w

        # Check that both methods produce the same set of indices (order may differ)
        set_d = Set([ado_idx_d[:, i] for i in 1:nado_d])
        set_w = Set([ado_idx_w[:, i] for i in 1:nado_w])
        @test set_d == set_w

        println("\n=== Comparison (nmode=$nmode, ndepth=$ndepth) ===")
        println("depth nado: ", nado_d)
        println("width nado: ", nado_w)
        println("Index sets match: ", set_d == set_w)
    end

    @testset "HEOMSystem width filtering API" begin
        H = ComplexF64[0 0; 0 1]
        expon = ComplexF64[1.0, 2.0]
        coeff = ComplexF64[1e-2, 1e-6]
        V = ComplexF64[1 0; 0 0]
        noise = NoiseExp([BathExp(expon, coeff, V)])

        system_unfiltered = HEOMSystem(H, noise, 4; hierarchy=:width)
        system_filtered = HEOMSystem(H, noise, 4;
            hierarchy=:width, tolerance=1e-3, filter=true)

        @test system_filtered.nado > 0
        @test system_filtered.nado < system_unfiltered.nado
    end
end
