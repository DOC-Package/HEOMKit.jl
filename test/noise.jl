using KaisouEOM
using Test

@testset "noise.jl" begin

    @testset "Bath construction (direct parameters)" begin
        # 直接パラメータを指定して Bath を作成
        expon = [1.0 + 0.5im, 2.0 - 0.3im]
        coeff = [0.1 + 0.2im, 0.3 - 0.1im]
        V = ComplexF64[0 1; 1 0]  # σx
        
        bath = BathExp(expon, coeff, V; add_conjugate=true)
        
        @test bath.nterms == 4  # 2 original + 2 conjugates
        @test size(bath.V) == (2, 2)
        
        # 複素共役が正しく追加されているか確認
        @test bath.expon[3] == conj(bath.expon[1])
        @test bath.expon[4] == conj(bath.expon[2])
        @test bath.coeff[3] == conj(bath.coeff[1])
        @test bath.coeff[4] == conj(bath.coeff[2])
        
        println("Bath (direct): nterms = $(bath.nterms)")
    end

    @testset "Bath construction (no conjugate)" begin
        expon = [1.0, 2.0]
        coeff = [0.1, 0.2]
        V = ComplexF64[1 0; 0 -1]  # σz
        
        bath = BathExp(expon, coeff, V; add_conjugate=false)
        
        @test bath.nterms == 2
        println("Bath (no conjugate): nterms = $(bath.nterms)")
    end

    @testset "Noise from single Bath" begin
        expon = [1.0 + 0.1im]
        coeff = [0.5 + 0.2im]
        V = ComplexF64[0 1; 1 0]
        
        bath = BathExp(expon, coeff, V; add_conjugate=true)
        noise = NoiseExp(bath)
        
        @test noise.nbath == 1
        @test noise.nterms == bath.nterms
        @test noise.nterms_bath == [bath.nterms]
        @test noise.jstart_bath == [1]
        @test length(noise.V) == 1
        
        println("\n=== Single Bath Noise ===")
        println("nbath = $(noise.nbath)")
        println("nterms = $(noise.nterms)")
        println("expon = $(noise.expon)")
        println("coeff = $(noise.coeff)")
    end

    @testset "Noise from multiple Baths" begin
        # Bath 1
        expon1 = [1.0 + 0.1im, 2.0]
        coeff1 = [0.3, 0.4 + 0.1im]
        V1 = ComplexF64[0 1; 1 0]  # σx
        bath1 = BathExp(expon1, coeff1, V1; add_conjugate=true)
        
        # Bath 2
        expon2 = [3.0]
        coeff2 = [0.5]
        V2 = ComplexF64[1 0; 0 -1]  # σz
        bath2 = BathExp(expon2, coeff2, V2; add_conjugate=true)
        
        noise = NoiseExp([bath1, bath2])
        
        @test noise.nbath == 2
        @test noise.nterms == bath1.nterms + bath2.nterms
        @test noise.nterms_bath == [bath1.nterms, bath2.nterms]
        @test noise.jstart_bath == [1, bath1.nterms + 1]
        @test length(noise.V) == 2
        
        # パラメータが正しく統合されているか
        @test noise.expon[1:bath1.nterms] == bath1.expon
        @test noise.expon[(bath1.nterms+1):end] == bath2.expon
        
        println("\n=== Multi-Bath Noise ===")
        println("nbath = $(noise.nbath)")
        println("nterms = $(noise.nterms)")
        println("nterms_bath = $(noise.nterms_bath)")
        println("jstart_bath = $(noise.jstart_bath)")
    end

    @testset "compute_heom_params" begin
        expon = [1.0 + 0.5im, 2.0 - 0.3im]
        coeff = [0.1 + 0.2im, 0.3 - 0.1im]
        V = ComplexF64[0 1; 1 0]
        
        bath = BathExp(expon, coeff, V; add_conjugate=true)
        noise = NoiseExp(bath)
        
        bk, ak, cb = compute_heom_params(noise)
        
        @test length(bk) == noise.nterms
        @test ak == 0.5
        @test length(cb) == noise.nterms
        @test all(bk .> 0)
        
        println("\n=== HEOM params ===")
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
