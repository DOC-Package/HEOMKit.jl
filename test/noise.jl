using KaisouEOM
using Test

@testset "noise.jl" begin

    @testset "Bath construction (direct parameters)" begin
        # 直接パラメータを指定して Bath を作成
        expon = [1.0 + 0.5im, 2.0 - 0.3im]
        coeff = [0.1 + 0.2im, 0.3 - 0.1im]
        V = ComplexF64[0 1; 1 0]  # σx
        
        bath = BathExp(expon, coeff, V)
        
        # BathExp stores the original parameters without expansion
        @test bath.nterms == 2
        @test size(bath.V) == (2, 2)
        @test bath.expon == ComplexF64[1.0 + 0.5im, 2.0 - 0.3im]
        @test bath.coeff == ComplexF64[0.1 + 0.2im, 0.3 - 0.1im]
        
        println("Bath (direct): nterms = $(bath.nterms)")
    end

    @testset "Bath construction (real exponents)" begin
        expon = [1.0, 2.0]
        coeff = [0.1, 0.2]
        V = ComplexF64[1 0; 0 -1]  # σz
        
        bath = BathExp(expon, coeff, V)
        
        @test bath.nterms == 2
        println("Bath (real exponents): nterms = $(bath.nterms)")
    end

    @testset "Noise from single Bath (complex exponent, no pair)" begin
        # Complex exponent without conjugate pair in BathExp
        # _expand_bcf_with_conjugate will add the conjugate mode
        expon = ComplexF64[1.0 + 0.1im]
        coeff = ComplexF64[0.5 + 0.2im]
        V = ComplexF64[0 1; 1 0]
        
        bath = BathExp(expon, coeff, V)
        noise = NoiseExp(bath)
        
        @test noise.nbath == 1
        @test noise.nterms == 2  # Original + added conjugate
        @test noise.nterms_bath == [2]
        @test noise.jstart_bath == [1]
        @test length(noise.V) == noise.nterms
        
        # For complex without pair: original gets c1=c, c2=0; conjugate gets c1=0, c2=c*
        # Positive imag comes first
        @test noise.γ[1] == expon[1]  # 1.0 + 0.1im (positive imag)
        @test noise.γ[2] == conj(expon[1])  # 1.0 - 0.1im (negative imag)
        @test noise.c1[1] == coeff[1]
        @test noise.c2[1] == 0.0
        @test noise.c1[2] == 0.0
        @test noise.c2[2] == conj(coeff[1])
        
        println("\n=== Single Bath Noise (no pair) ===")
        println("nbath = $(noise.nbath)")
        println("nterms = $(noise.nterms)")
        println("γ = $(noise.γ)")
        println("c1 = $(noise.c1)")
        println("c2 = $(noise.c2)")
    end

    @testset "Noise from multiple Baths" begin
        # Bath 1: one complex (will be expanded to 2), one real (stays 1)
        expon1 = ComplexF64[1.0 + 0.1im, 2.0]
        coeff1 = ComplexF64[0.3, 0.4 + 0.1im]
        V1 = ComplexF64[0 1; 1 0]  # σx
        bath1 = BathExp(expon1, coeff1, V1)
        
        # Bath 2: one real (stays 1)
        expon2 = ComplexF64[3.0]
        coeff2 = ComplexF64[0.5]
        V2 = ComplexF64[1 0; 0 -1]  # σz
        bath2 = BathExp(expon2, coeff2, V2)
        
        noise = NoiseExp([bath1, bath2])
        
        @test noise.nbath == 2
        # Bath1: 1 complex (2 after expansion) + 1 real (1) = 3
        # Bath2: 1 real = 1
        @test noise.nterms_bath == [3, 1]
        @test noise.nterms == 4
        @test noise.jstart_bath == [1, 4]
        @test length(noise.V) == noise.nterms
        
        println("\n=== Multi-Bath Noise ===")
        println("nbath = $(noise.nbath)")
        println("nterms = $(noise.nterms)")
        println("nterms_bath = $(noise.nterms_bath)")
        println("jstart_bath = $(noise.jstart_bath)")
        println("γ = $(noise.γ)")
        println("c1 = $(noise.c1)")
        println("c2 = $(noise.c2)")
    end

    @testset "c1_c2 for conjugate pair in input" begin
        # BCF decomposition that already contains conjugate pairs
        # expon[1] = 1.0 + 0.5im, expon[2] = 1.0 - 0.5im (conjugate pair)
        expon = ComplexF64[1.0 + 0.5im, 1.0 - 0.5im]
        # Deliberately unequal coefficients distinguish the two sides of the
        # finite-temperature correlation function.
        coeff = ComplexF64[0.3 + 0.2im, 0.07 - 0.04im]
        V = ComplexF64[0 1; 1 0]
        
        bath = BathExp(expon, coeff, V)
        noise = NoiseExp(bath)
        
        @test noise.nterms == 2  # No expansion, pair already exists
        
        # c1 reconstructs C(t), while c2 reconstructs C(t)* after matching
        # each exponent with the coefficient of its conjugate exponent.
        ca, cb = coeff[1], coeff[2]
        γa, γb = expon[1], expon[2]
        
        # Positive imag first
        @test noise.γ[1] == γa  # 1.0 + 0.5im
        @test noise.γ[2] == γb  # 1.0 - 0.5im
        @test noise.c1 == ComplexF64[ca, cb]
        @test noise.c2 == ComplexF64[conj(cb), conj(ca)]
        @test noise.abs_coeff ≈ abs.(noise.c1 .+ noise.c2)

        for time in (0.0, 0.37, 1.1)
            correlation = sum(coeff .* exp.(-expon .* time))
            reconstructed_c1 = sum(noise.c1 .* exp.(-noise.γ .* time))
            reconstructed_c2 = sum(noise.c2 .* exp.(-noise.γ .* time))
            @test reconstructed_c1 ≈ correlation
            @test reconstructed_c2 ≈ conj(correlation)
        end
        
        println("\n=== c1_c2 for conjugate pair ===")
        println("γ = $(noise.γ)")
        println("c1 = $(noise.c1)")
        println("c2 = $(noise.c2)")
    end

    @testset "c1_c2 for real exponents" begin
        # BCF decomposition with real exponents (self-conjugate)
        expon = ComplexF64[1.0, 2.0]
        coeff = ComplexF64[0.5 + 0.1im, 0.3 - 0.2im]
        V = ComplexF64[0 1; 1 0]
        
        bath = BathExp(expon, coeff, V)
        noise = NoiseExp(bath)
        
        @test noise.nterms == 2  # Real exponents don't expand
        # For real exponents: c1 = c, c2 = c*
        # Note: order may be different due to sorting by |c1+c2|/Re(γ)
        # Check that each mode has correct c1, c2
        for k in 1:2
            # Find which original coeff this corresponds to
            γ_k = real(noise.γ[k])
            if abs(γ_k - 1.0) < 1e-10
                @test noise.c1[k] ≈ coeff[1]
                @test noise.c2[k] ≈ conj(coeff[1])
            else
                @test noise.c1[k] ≈ coeff[2]
                @test noise.c2[k] ≈ conj(coeff[2])
            end
        end
        
        println("\n=== c1_c2 for real exponents ===")
        println("γ = $(noise.γ)")
        println("c1 = $(noise.c1)")
        println("c2 = $(noise.c2)")
    end

    @testset "compute_heom_params" begin
        expon = ComplexF64[1.0 + 0.5im, 2.0 - 0.3im]
        coeff = ComplexF64[0.1 + 0.2im, 0.3 - 0.1im]
        V = ComplexF64[0 1; 1 0]
        
        bath = BathExp(expon, coeff, V)
        noise = NoiseExp(bath)
        
        bk, ak, cb = compute_heom_params(noise)
        
        @test length(bk) == noise.nterms
        @test ak == 0.5
        @test length(cb) == noise.nterms
        @test all(bk .> 0)
        
        println("\n=== HEOM params ===")
        println("nterms = $(noise.nterms)")
        println("bk = $bk")
        println("ak = $ak")
        println("cb = $cb")
    end

    @testset "Physical constants" begin
        # ボルツマン定数の確認
        @test kB ≈ 0.695034800 atol=1e-6
        
        # icm2ifs の確認 (2π × c × 1e-15)
        c_light = 2.99792458e10  # cm/s
        expected_icm2ifs = 2π * c_light * 1e-15
        @test icm2ifs ≈ expected_icm2ifs atol=1e-10
        
        println("\n=== Physical constants ===")
        println("kB = $kB cm⁻¹/K")
        println("icm2ifs = $icm2ifs")
    end

end
