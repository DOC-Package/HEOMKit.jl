using KaisouEOM
using KaisouEOM: icm2ifs, kB, build_tridiagonal_D  # 定数をインポート
using QFiND
using ExpFit
using LinearAlgebra
using Test

@testset "HEOM Integration Test" begin
    
    @testset "2-level system with Drude bath" begin
        println("\n=== 2-level system with Drude bath ===")
        
        # ハミルトニアン (2準位系) [cm⁻¹]
        ε = 0.0      # エネルギー差
        Δ = 100.0    # トンネル結合
        H = [ε/2 Δ; Δ -ε/2]
        
        println("Hamiltonian H:")
        display(H)
        
        # 熱浴パラメータ
        λ = 100.0    # 再構成エネルギー [cm⁻¹]
        γ = 50.0     # カットオフ周波数 [cm⁻¹]
        T = 300.0    # 温度 [K]
        
        # 相互作用演算子 (σz)
        V = ComplexF64[1 0; 0 -1]
        
        # ノイズパラメータを直接指定してテスト
        # (QFiND の SpectralDensity を使わずにシンプルなテスト)
        expon = ComplexF64[γ * icm2ifs]  # [1/fs]
        coeff = ComplexF64[λ * γ / (kB * T) * icm2ifs^2]  # 高温極限の近似
        
        bath = Bath(expon, coeff, V; add_conjugate=true)
        noise = Noise(bath)
        
        println("\nNoise parameters:")
        println("  nterms = $(noise.nterms)")
        println("  expon = $(noise.expon)")
        println("  coeff = $(noise.coeff)")
        
        # 単位変換されたハミルトニアン [1/fs]
        H_fs = H * icm2ifs
        
        # HEOM システム構築
        ndepth = 3
        system = HEOMSystem(H_fs, noise, ndepth; hierarchy=:depth)
        
        println("\nHEOM System:")
        println("  nado = $(system.nado)")
        println("  ndim = $(system.ndim)")
        println("  ndim² = $(system.ndim2)")
        
        @test system.nado == binomial(ndepth + noise.nterms, noise.nterms)
        @test system.ndim == 2
        @test system.ndim2 == 4
        
        # 初期条件: |1⟩⟨1| (状態1に局在)
        P0 = initial_ado(system, 1)
        
        println("\nInitial ADO:")
        println("  size = $(size(P0))")
        println("  P0[:, 1] = $(P0[:, 1])")  # ρ₀
        
        @test size(P0) == (system.ndim2, system.nado)
        @test P0[1, 1] ≈ 1.0  # ρ₁₁ = 1
        @test P0[4, 1] ≈ 0.0  # ρ₂₂ = 0
        
        # 時間発展 (短い時間でテスト)
        dt = 1.0  # [fs]
        t_end = 100.0  # [fs]
        
        println("\nTime evolution:")
        println("  dt = $dt fs")
        println("  t_end = $t_end fs")
        
        times, pops = evolve(system, P0, (0.0, t_end), dt)
        
        println("\nResults:")
        println("  Number of time steps: $(length(times))")
        println("  Initial population: ρ₁₁=$(pops[1,1]), ρ₂₂=$(pops[2,1])")
        println("  Final population: ρ₁₁=$(pops[1,end]), ρ₂₂=$(pops[2,end])")
        
        # 基本的なテスト
        @test length(times) == Int(t_end / dt) + 1
        @test times[1] ≈ 0.0
        @test times[end] ≈ t_end
        
        # 初期状態の確認
        @test pops[1, 1] ≈ 1.0 atol=1e-10
        @test pops[2, 1] ≈ 0.0 atol=1e-10
        
        # 確率保存の確認 (トレースが1)
        for i in 1:length(times)
            total_pop = pops[1, i] + pops[2, i]
            @test total_pop ≈ 1.0 atol=1e-6
        end
        
        # 何らかのダイナミクスが起きているか確認
        @test pops[1, end] != pops[1, 1]  # 時間変化がある
        
        println("\n=== Test passed! ===")
    end
    
    @testset "Liouville operator test" begin
        println("\n=== Liouville operator test ===")
        
        # シンプルな2準位系
        H = ComplexF64[0 1; 1 0] * icm2ifs
        V = ComplexF64[1 0; 0 -1]
        
        expon = ComplexF64[0.01]
        coeff = ComplexF64[0.001]
        bath = Bath(expon, coeff, V; add_conjugate=true)
        noise = Noise(bath)
        
        system = HEOMSystem(H, noise, 2; hierarchy=:depth)

        # 初期状態
        P0 = initial_ado(system, 1)

        # Liouville 演算子を適用
        dP = liouville(P0, system)
        
        println("Liouville output size: $(size(dP))")
        println("dP[:, 1] = $(dP[:, 1])")
        
        @test size(dP) == size(P0)
        @test !all(dP .== 0)  # 非ゼロの出力があるはず
        
        println("=== Test passed! ===")
    end
    
    @testset "LSRK4 integrator test" begin
        println("\n=== LSRK4 integrator test ===")
        
        H = ComplexF64[0 1; 1 0] * icm2ifs
        V = ComplexF64[1 0; 0 -1]
        
        expon = ComplexF64[0.01]
        coeff = ComplexF64[0.001]
        bath = Bath(expon, coeff, V; add_conjugate=true)
        noise = Noise(bath)
        
        system = HEOMSystem(H, noise, 2; hierarchy=:depth)
        P0 = initial_ado(system, 1)

        # 1ステップ進める
        P1 = lsrk4(P0, 1.0, system)
        
        println("P0[:, 1] = $(P0[:, 1])")
        println("P1[:, 1] = $(P1[:, 1])")
        
        @test size(P1) == size(P0)
        @test P1 != P0  # 変化がある
        
        # トレース保存の確認
        trace_P0 = real(P0[1, 1] + P0[4, 1])
        trace_P1 = real(P1[1, 1] + P1[4, 1])
        println("Trace P0 = $trace_P0")
        println("Trace P1 = $trace_P1")
        @test trace_P1 ≈ trace_P0 atol=1e-6
        
        println("=== Test passed! ===")
    end

    @testset "Ohmic bath with ESPRIT fitting" begin
        println("\n=== Ohmic bath with ESPRIT fitting ===")
        
        # Ohmic型スペクトル密度（PowerLawExpSD, s=1）
        s = 1.0        # Ohmic
        γ = 100.0      # カットオフ周波数 [cm⁻¹]
        λ = 50.0       # 再構成エネルギー [cm⁻¹]
        T = 300.0      # 温度 [K]
        
        sd = PowerLawExpSD(s, γ; reorgene=λ)
        bcf = BosonicBCF(sd, T)
        
        println("Spectral density: PowerLawExpSD (Ohmic)")
        println("  s = $s, γ = $γ cm⁻¹, λ = $λ cm⁻¹, T = $T K")
        
        # ESPRITで相関関数を指数和近似
        tmin = 0.0
        tmax = 500.0   # [fs]
        nsamples = 200
        eps = 1e-6
        
        # 相関関数のサンプリング
        dt = (tmax - tmin) / (nsamples - 1)
        t_samples = range(tmin, tmax, length=nsamples)
        bcf_samples = [bcf(t) for t in t_samples]
        
        println("\nBCF sampling:")
        println("  tmin = $tmin fs, tmax = $tmax fs")
        println("  nsamples = $nsamples, dt = $dt fs")
        println("  bcf(0) = $(bcf_samples[1])")
        
        # ESPRITで指数和近似
        ef = ExpFit.esprit(bcf_samples, dt, eps)
        
        println("\nESPRIT fitting result:")
        println("  Number of terms: $(length(ef.expon))")
        
        # フィッティング精度の確認
        bcf_fit = [ef(t) for t in t_samples]
        fit_error = norm(bcf_fit .- bcf_samples) / norm(bcf_samples)
        println("  Relative fitting error: $fit_error")
        @test fit_error < 1e-4
        
        # ExpFitの結果をKaisouEOMの熱浴パラメータに変換
        # ExpFitの指数: f(t) = Σ cₖ exp(-aₖ t)
        # KaisouEOMの熱浴: C(t) = Σ cₖ exp(-γₖ t)
        # → gamk = expon, ck = coeff
        gamk = ef.expon
        ck = ef.coeff
        
        println("\nBath parameters from ESPRIT:")
        println("  Number of modes: $(length(gamk))")
        for i in 1:min(5, length(gamk))
            println("    γ[$i] = $(gamk[i]), c[$i] = $(ck[i])")
        end
        if length(gamk) > 5
            println("    ...")
        end
        
        # 熱浴の構築（複素共役ペアは既にESPRITの結果に含まれているはず）
        V = ComplexF64[1 0; 0 -1]  # σz
        bath = Bath(gamk, ck, V; add_conjugate=false)
        noise = Noise(bath)
        
        println("\nNoise structure:")
        println("  nterms = $(noise.nterms)")
        
        # 相関関数の再構成確認
        bcf_reconstructed = [sum(ck .* exp.(-gamk .* t)) for t in t_samples]
        recon_error = norm(bcf_reconstructed .- bcf_samples) / norm(bcf_samples)
        println("  Reconstruction error: $recon_error")
        @test recon_error < 1e-4
        
        # ハミルトニアン（2準位系）
        ε = 0.0      # エネルギー差 [cm⁻¹]
        Δ = 100.0    # トンネル結合 [cm⁻¹]
        H = [ε/2 Δ; Δ -ε/2] * icm2ifs  # [1/fs]
        
        # HEOMシステム構築
        ndepth = 2
        system = HEOMSystem(H, noise, ndepth; hierarchy=:depth)
        
        println("\nHEOM System:")
        println("  nado = $(system.nado)")
        
        # 初期条件と時間発展
        P0 = initial_ado(system, 1)
        t_end = 100.0  # [fs]
        dt_evolve = 1.0  # [fs]
        
        times, pops = evolve(system, P0, (0.0, t_end), dt_evolve)
        
        println("\nTime evolution:")
        println("  Initial: ρ₁₁=$(pops[1,1]), ρ₂₂=$(pops[2,1])")
        println("  Final: ρ₁₁=$(pops[1,end]), ρ₂₂=$(pops[2,end])")
        
        # 確率保存の確認
        for i in 1:length(times)
            total_pop = pops[1, i] + pops[2, i]
            @test total_pop ≈ 1.0 atol=1e-5
        end
        
        println("\n=== Test passed! ===")
    end
    
end


@testset "HSEOM Integration Test" begin
    
    @testset "HSEOM bra/ket simultaneous evolution" begin
        println("\n=== HSEOM bra/ket simultaneous evolution ===")
        
        # ハミルトニアン (2準位系)
        Δ = 100.0  # トンネル結合 [cm⁻¹]
        H = ComplexF64[0 Δ; Δ 0] * icm2ifs  # [1/fs]
        
        # 相互作用演算子 (σz)
        V = ComplexF64[1 0; 0 -1]
        
        # ノイズパラメータ
        expon = ComplexF64[0.01 + 0.001im]  # [1/fs]
        coeff = ComplexF64[0.001 + 0.0001im]  # 係数
        bath = Bath(expon, coeff, V; add_conjugate=true)
        noise = Noise(bath)
        
        println("Noise parameters:")
        println("  nterms = $(noise.nterms)")
        
        # D 行列（三重対角）
        gamma_c = 0.05
        D = build_tridiagonal_D(noise.nterms, gamma_c)
        
        # HSEOM システム構築
        ndepth = 3
        system = HSEOMSystem(H, noise, D, ndepth)
        
        println("\nHSEOM System:")
        println("  nadw = $(system.nadw)")
        println("  ndim = $(system.ndim)")
        println("  nterms = $(system.nterms)")
        
        @test system.nadw == binomial(ndepth + noise.nterms, noise.nterms)
        @test system.ndim == 2
        
        # 初期条件: |1⟩ (bra と ket 両方)
        Pb0 = initial_adw(system, 1)
        Pk0 = initial_adw(system, 1)
        
        println("\nInitial ADW:")
        println("  size = $(size(Pk0))")
        println("  Pk0[:, 1] = $(Pk0[:, 1])")
        
        @test size(Pk0) == (system.ndim, system.nadw)
        @test Pk0[1, 1] ≈ 1.0  # |1⟩
        @test Pk0[2, 1] ≈ 0.0
        
        # 時間発展
        dt = 1.0  # [fs]
        t_end = 100.0  # [fs]
        
        println("\nTime evolution:")
        println("  dt = $dt fs")
        println("  t_end = $t_end fs")
        
        times, pops = evolve(system, Pb0, Pk0, (0.0, t_end), dt)
        
        println("\nResults:")
        println("  Number of time steps: $(length(times))")
        println("  Initial population: p₁=$(pops[1,1]), p₂=$(pops[2,1])")
        println("  Final population: p₁=$(pops[1,end]), p₂=$(pops[2,end])")
        
        # 基本的なテスト
        @test length(times) == Int(t_end / dt) + 1
        @test times[1] ≈ 0.0
        @test times[end] ≈ t_end
        
        # 初期状態の確認
        @test pops[1, 1] ≈ 1.0 atol=1e-10
        @test pops[2, 1] ≈ 0.0 atol=1e-10
        
        # 何らかのダイナミクスが起きているか確認
        @test pops[1, end] != pops[1, 1]  # 時間変化がある
        
        println("\n=== Test passed! ===")
    end
    
    @testset "HSEOM population calculation" begin
        println("\n=== HSEOM population calculation ===")
        
        # シンプルな系でポピュレーション計算を確認
        H = ComplexF64[0 100; 100 0] * icm2ifs
        V = ComplexF64[1 0; 0 -1]
        
        expon = ComplexF64[0.01]
        coeff = ComplexF64[0.001]
        bath = Bath(expon, coeff, V; add_conjugate=true)
        noise = Noise(bath)
        
        D = build_tridiagonal_D(noise.nterms, 0.05)
        system = HSEOMSystem(H, noise, D, 3)
        
        # |1⟩ で始める → ρ₁₁ = 1, ρ₂₂ = 0
        Pb0 = initial_adw(system, 1)
        Pk0 = initial_adw(system, 1)
        
        # t=0 でのポピュレーション
        ndim = system.ndim
        nadw = system.nadw
        
        p1_t0 = real(sum(Pk0[1, n] * conj(Pb0[1, n]) for n in 1:nadw))
        p2_t0 = real(sum(Pk0[2, n] * conj(Pb0[2, n]) for n in 1:nadw))
        
        println("t=0: p₁ = $p1_t0, p₂ = $p2_t0, total = $(p1_t0 + p2_t0)")
        
        @test p1_t0 ≈ 1.0
        @test p2_t0 ≈ 0.0
        
        # |2⟩ で始める → ρ₁₁ = 0, ρ₂₂ = 1
        Pb0_2 = initial_adw(system, 2)
        Pk0_2 = initial_adw(system, 2)
        
        p1_state2 = real(sum(Pk0_2[1, n] * conj(Pb0_2[1, n]) for n in 1:nadw))
        p2_state2 = real(sum(Pk0_2[2, n] * conj(Pb0_2[2, n]) for n in 1:nadw))
        
        println("State |2⟩: p₁ = $p1_state2, p₂ = $p2_state2")
        
        @test p1_state2 ≈ 0.0
        @test p2_state2 ≈ 1.0
        
        # 重ね合わせ |+⟩ = (|1⟩ + |2⟩)/√2 → ρ₁₁ = ρ₂₂ = 0.5
        psi_plus = ComplexF64[1/sqrt(2), 1/sqrt(2)]
        Pb0_plus = initial_adw(system, psi_plus)
        Pk0_plus = initial_adw(system, psi_plus)
        
        p1_plus = real(sum(Pk0_plus[1, n] * conj(Pb0_plus[1, n]) for n in 1:nadw))
        p2_plus = real(sum(Pk0_plus[2, n] * conj(Pb0_plus[2, n]) for n in 1:nadw))
        
        println("State |+⟩: p₁ = $p1_plus, p₂ = $p2_plus")
        
        @test p1_plus ≈ 0.5 atol=1e-10
        @test p2_plus ≈ 0.5 atol=1e-10
        
        println("\n=== Test passed! ===")
    end
    
end
