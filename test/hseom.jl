# HSEOM tests
using KaisouEOM
using KaisouEOM: icm2ifs, build_tridiagonal_D
using LinearAlgebra
using Test

@testset "HSEOM" begin

    @testset "HSEOMSystem construction" begin
        println("\n=== HSEOMSystem construction ===")
        
        # 2準位系ハミルトニアン
        H = ComplexF64[0 100; 100 0] * icm2ifs
        V = ComplexF64[1 0; 0 -1]
        
        # ノイズパラメータ (complex without pair → will be expanded to 2 modes)
        expon = ComplexF64[0.01 + 0.001im]
        coeff = ComplexF64[0.001 + 0.0001im]
        bath = BathExp(expon, coeff, V)
        noise = NoiseExp(bath)
        
        # HSEOMシステム構築
        ndepth = 3
        gamma_c = 0.05  # カットオフ周波数
        D = build_tridiagonal_D(noise.nterms, gamma_c)
        system = HSEOMSystem(H, noise, D, ndepth)
        
        println("ndim = $(system.ndim)")
        println("nadw = $(system.nadw)")
        println("D matrix size = $(size(system.D))")
        
        # 基本的なテスト
        @test system.ndim == 2
        @test system.nadw == binomial(ndepth + noise.nterms, noise.nterms)
        @test size(system.D) == (noise.nterms, noise.nterms)
        
        # H と V が正しく保持されているか
        @test size(system.H) == (2, 2)
        @test length(system.V) == 1
        @test size(system.V[1]) == (2, 2)
        
        # 2モード変化インデックスが存在するか（3次元配列）
        @test size(system.idx_minus_plus) == (noise.nterms, noise.nterms, system.nadw)
        @test size(system.idx_plus_minus) == (noise.nterms, noise.nterms, system.nadw)
        
        println("=== Test passed! ===")
    end
    
    @testset "initial_adw" begin
        println("\n=== initial_adw ===")
        
        H = ComplexF64[0 100; 100 0] * icm2ifs
        V = ComplexF64[1 0; 0 -1]
        
        # Real exponent → 1 mode (no expansion)
        expon = ComplexF64[0.01]
        coeff = ComplexF64[0.001]
        bath = BathExp(expon, coeff, V)
        noise = NoiseExp(bath)
        
        D = build_tridiagonal_D(noise.nterms, 0.05)
        system = HSEOMSystem(H, noise, D, 3)
        
        # 状態1から始める
        P0 = initial_adw(system, 1)
        
        println("size(P0) = $(size(P0))")
        println("P0[:, 1] = $(P0[:, 1])")
        
        @test size(P0) == (system.ndim, system.nadw)
        @test P0[1, 1] ≈ 1.0
        @test P0[2, 1] ≈ 0.0
        @test all(P0[:, 2:end] .== 0.0)  # 他のADWはゼロ
        
        # カスタム波動関数
        psi0 = ComplexF64[1/sqrt(2), 1/sqrt(2)]
        P0_custom = initial_adw(system, psi0)
        
        @test P0_custom[:, 1] ≈ psi0
        @test norm(P0_custom[:, 1]) ≈ 1.0
        
        println("=== Test passed! ===")
    end
    
    @testset "liouville_ket!" begin
        println("\n=== liouville_ket! ===")
        
        H = ComplexF64[0 100; 100 0] * icm2ifs
        V = ComplexF64[1 0; 0 -1]
        
        # Real exponent → 1 mode
        expon = ComplexF64[0.01]
        coeff = ComplexF64[0.001]
        bath = BathExp(expon, coeff, V)
        noise = NoiseExp(bath)
        
        D = build_tridiagonal_D(noise.nterms, 0.05)
        system = HSEOMSystem(H, noise, D, 3)
        
        P0 = initial_adw(system, 1)
        dP = similar(P0)
        
        # Liouville演算子を適用
        liouville_ket!(dP, P0, system)
        
        println("dP[:, 1] = $(dP[:, 1])")
        
        # 結果が非ゼロであることを確認
        @test !all(dP .== 0.0)
        
        # システム項の確認: -i * H * |1⟩ = -i * [0, 100] * icm2ifs
        expected_system = -1.0im * H * P0[:, 1]
        
        # liouville_ket（新配列版）も動作確認
        dP2 = liouville_ket(P0, system)
        @test dP ≈ dP2
        
        println("=== Test passed! ===")
    end
    
    @testset "liouville_bra!" begin
        println("\n=== liouville_bra! ===")
        
        H = ComplexF64[0 100; 100 0] * icm2ifs
        V = ComplexF64[1 0; 0 -1]
        
        # Real exponent → 1 mode
        expon = ComplexF64[0.01]
        coeff = ComplexF64[0.001]
        bath = BathExp(expon, coeff, V)
        noise = NoiseExp(bath)
        
        D = build_tridiagonal_D(noise.nterms, 0.05)
        system = HSEOMSystem(H, noise, D, 3)
        
        P0 = initial_adw(system, 1)
        dP = similar(P0)
        
        # Liouville演算子を適用
        liouville_bra!(dP, P0, system)
        
        println("dP[:, 1] = $(dP[:, 1])")
        
        # 結果が非ゼロであることを確認
        @test !all(dP .== 0.0)
        
        # liouville_bra（新配列版）も動作確認
        dP2 = liouville_bra(P0, system)
        @test dP ≈ dP2
        
        println("=== Test passed! ===")
    end
    
    @testset "2-mode index maps (general D)" begin
        println("\n=== 2-mode index maps (general D) ===")
        
        H = ComplexF64[0 100; 100 0] * icm2ifs
        V = ComplexF64[1 0; 0 -1]
        
        # 3つのモードを持つ熱浴 (real exponents → no expansion)
        expon = ComplexF64[0.01, 0.02, 0.03]
        coeff = ComplexF64[0.001, 0.002, 0.003]
        bath = BathExp(expon, coeff, V)
        noise = NoiseExp(bath)
        
        # 一般の D 行列（ここでは三重対角）
        D = build_tridiagonal_D(noise.nterms, 0.05)
        system = HSEOMSystem(H, noise, D, 3)
        
        println("nterms = $(noise.nterms)")
        println("nadw = $(system.nadw)")
        
        # 3次元配列インデックスマップのサイズ確認
        @test size(system.idx_minus_plus) == (3, 3, system.nadw)
        @test size(system.idx_plus_minus) == (3, 3, system.nadw)
        
        # k=ℓ（対角成分）は常に -1（単一モード変化は別で扱う）
        for n in 1:system.nadw
            for k in 1:noise.nterms
                @test system.idx_minus_plus[k, k, n] == -1
                @test system.idx_plus_minus[k, k, n] == -1
            end
        end
        
        println("=== Test passed! ===")
    end
    
    @testset "HSEOM vs HEOM dimensions" begin
        println("\n=== HSEOM vs HEOM dimensions ===")
        
        H = ComplexF64[0 100; 100 0] * icm2ifs
        V = ComplexF64[1 0; 0 -1]
        
        # Real exponent → 1 mode
        expon = ComplexF64[0.01]
        coeff = ComplexF64[0.001]
        bath = BathExp(expon, coeff, V)
        noise = NoiseExp(bath)
        
        ndepth = 3
        
        # HEOM（密度行列）
        heom = HEOMSystem(H, noise, ndepth)
        
        # HSEOM（波動関数）
        D = build_tridiagonal_D(noise.nterms, 0.05)
        hseom = HSEOMSystem(H, noise, D, ndepth)
        
        println("HEOM:  ndim=$(heom.ndim), ndim2=$(heom.ndim2), nado=$(heom.nado)")
        println("HSEOM: ndim=$(hseom.ndim), nadw=$(hseom.nadw)")
        
        # HEOM は Liouville 空間 (ndim²)
        @test heom.ndim2 == heom.ndim^2
        
        # HSEOM は Hilbert 空間 (ndim)
        @test hseom.ndim == heom.ndim
        
        # ADO/ADW の数は同じ（階層構造は同じ）
        @test heom.nado == hseom.nadw
        
        # 初期条件の次元
        P_heom = initial_ado(heom, 1)
        P_hseom = initial_adw(hseom, 1)
        
        @test size(P_heom, 1) == heom.ndim2  # ρ はベクトル化 (ndim²)
        @test size(P_hseom, 1) == hseom.ndim  # ψ はベクトル (ndim)
        
        println("=== Test passed! ===")
    end

    @testset "HSEOM evolve (bra/ket simultaneous)" begin
        println("\n=== HSEOM evolve (bra/ket simultaneous) ===")
        
        # 2準位系ハミルトニアン
        H = ComplexF64[0 100; 100 0] * icm2ifs
        V = ComplexF64[1 0; 0 -1]
        
        # Complex without pair → expanded to 2 modes
        expon = ComplexF64[0.01 + 0.001im]
        coeff = ComplexF64[0.001 + 0.0001im]
        bath = BathExp(expon, coeff, V)
        noise = NoiseExp(bath)
        
        # HSEOM システム
        D = build_tridiagonal_D(noise.nterms, 0.05)
        system = HSEOMSystem(H, noise, D, 3)
        
        # 初期条件: |1⟩ (bra と ket 両方)
        Pb0 = initial_adw(system, 1)
        Pk0 = initial_adw(system, 1)
        
        # 時間発展
        dt = 1.0
        t_end = 50.0
        times, pops = evolve(system, Pb0, Pk0, (0.0, t_end), dt)
        
        println("Initial: p₁=$(pops[1,1]), p₂=$(pops[2,1])")
        println("Final:   p₁=$(pops[1,end]), p₂=$(pops[2,end])")
        
        # 基本テスト
        @test length(times) == Int(t_end / dt) + 1
        @test times[1] ≈ 0.0
        @test times[end] ≈ t_end
        
        # 初期状態
        @test pops[1, 1] ≈ 1.0 atol=1e-10
        @test pops[2, 1] ≈ 0.0 atol=1e-10
        
        # ダイナミクスが起きている
        @test pops[1, end] != pops[1, 1]
        
        println("=== Test passed! ===")
    end

    @testset "HSEOM population calculation" begin
        println("\n=== HSEOM population calculation ===")
        
        H = ComplexF64[0 100; 100 0] * icm2ifs
        V = ComplexF64[1 0; 0 -1]
        
        # Real exponent → 1 mode
        expon = ComplexF64[0.01]
        coeff = ComplexF64[0.001]
        bath = BathExp(expon, coeff, V)
        noise = NoiseExp(bath)
        
        D = build_tridiagonal_D(noise.nterms, 0.05)
        system = HSEOMSystem(H, noise, D, 3)
        
        nadw = system.nadw
        
        # |1⟩ → p₁=1, p₂=0
        Pb1 = initial_adw(system, 1)
        Pk1 = initial_adw(system, 1)
        p1 = real(sum(Pk1[1, n] * conj(Pb1[1, n]) for n in 1:nadw))
        p2 = real(sum(Pk1[2, n] * conj(Pb1[2, n]) for n in 1:nadw))
        
        println("State |1⟩: p₁=$p1, p₂=$p2")
        @test p1 ≈ 1.0
        @test p2 ≈ 0.0
        
        # |2⟩ → p₁=0, p₂=1
        Pb2 = initial_adw(system, 2)
        Pk2 = initial_adw(system, 2)
        p1 = real(sum(Pk2[1, n] * conj(Pb2[1, n]) for n in 1:nadw))
        p2 = real(sum(Pk2[2, n] * conj(Pb2[2, n]) for n in 1:nadw))
        
        println("State |2⟩: p₁=$p1, p₂=$p2")
        @test p1 ≈ 0.0
        @test p2 ≈ 1.0
        
        # |+⟩ = (|1⟩ + |2⟩)/√2 → p₁=p₂=0.5
        psi_plus = ComplexF64[1/sqrt(2), 1/sqrt(2)]
        Pb_plus = initial_adw(system, psi_plus)
        Pk_plus = initial_adw(system, psi_plus)
        p1 = real(sum(Pk_plus[1, n] * conj(Pb_plus[1, n]) for n in 1:nadw))
        p2 = real(sum(Pk_plus[2, n] * conj(Pb_plus[2, n]) for n in 1:nadw))
        
        println("State |+⟩: p₁=$p1, p₂=$p2")
        @test p1 ≈ 0.5 atol=1e-10
        @test p2 ≈ 0.5 atol=1e-10
        
        println("=== Test passed! ===")
    end

end
