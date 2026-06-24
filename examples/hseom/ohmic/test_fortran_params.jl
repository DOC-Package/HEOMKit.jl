"""
HSEOM Test: Fortran と同じパラメータで動作確認

Fortran setpara から推定されるパラメータ:
- NL = 2 (2準位系)
- gamc = γc (カットオフ周波数、Ω に相当)
- Jmax = n_terms (Chebyshev 項数)
- ck = 展開係数
"""

using KaisouEOM
using KaisouEOM: icm2ifs, kB
using LinearAlgebra

println("=" ^ 60)
println("HSEOM Test: Fortran Parameters")
println("=" ^ 60)

# =============================================
# Fortran と同じパラメータ設定
# =============================================

# システムパラメータ (Fortran setpara 相当)
NL = 2           # ヒルベルト空間次元
Δ = 100.0        # トンネル結合 [cm⁻¹]
H = ComplexF64[0 Δ; Δ 0] * icm2ifs  # [1/fs]

# 熱浴パラメータ
gamc = 50.0 * icm2ifs   # γc [1/fs] - Ω に相当
Jmax = 5                 # Chebyshev 項数 (小さく)

# 相互作用演算子
V = ComplexF64[1 0; 0 -1]

# Chebyshev 展開係数 (簡略化: 実数のみ)
# 実際の値は QFiND から得るが、ここではテスト用の値
ck = zeros(ComplexF64, Jmax)
ck[1] = 0.002   # c₀
ck[2] = 0.0005  # c₁
ck[3] = 0.0003  # c₂
ck[4] = 0.0002  # c₃
ck[5] = 0.0001  # c₄

println("\nParameters:")
println("  NL (ndim) = $NL")
println("  Δ = $Δ cm⁻¹")
println("  γc = $(gamc / icm2ifs) cm⁻¹ = $gamc [1/fs]")
println("  Jmax = $Jmax")
println("  ck = $ck")

# =============================================
# D 行列の構築 (Fortran と同じ三重対角)
# =============================================
# Fortran での D 行列項:
#   j=0: -nvec(0,n) * gamc * P(:, n-e₀+e₁)
#   j=1..Jmax-1: +0.5 * nvec(j,n) * gamc * (P(:,n-eⱼ+eⱼ₋₁) - P(:,n-eⱼ+eⱼ₊₁))
#
# これは D 行列で表すと:
#   D[1,2] = -gamc (j=0 は特別)
#   D[k,k-1] = +0.5*gamc (k≥2)
#   D[k,k+1] = -0.5*gamc (k≤Jmax-1)

D_matrix = zeros(ComplexF64, Jmax, Jmax)

# j=0 (k=1): D[1,2] = -gamc
if Jmax > 1
    D_matrix[1, 2] = -gamc
end

# j=1 to Jmax-1 (k=2 to Jmax)
for k in 2:Jmax
    if k > 1
        D_matrix[k, k-1] = 0.5 * gamc  # prev
    end
    if k < Jmax
        D_matrix[k, k+1] = -0.5 * gamc  # next
    end
end

println("\nD matrix:")
display(D_matrix)

# =============================================
# HSEOM システム構築
# =============================================

# Noise 構造体の作成
expon = zeros(ComplexF64, Jmax)  # D 行列で時間発展を扱うので使わない
coeff = ck

bath = BathExp(expon, coeff, V)
noise = NoiseExp(bath)

# HSEOM システム
ndepth = 3
system = HSEOMSystem(H, noise, D_matrix, ndepth; hierarchy=:depth)

println("\nHSEOM System:")
println("  Hierarchy depth: $ndepth")
println("  Number of ADWs: $(system.nadw)")
println("  nterms: $(system.nterms)")

# =============================================
# 時間発展
# =============================================

# 初期条件
Pb0 = initial_adw(system, 1)
Pk0 = initial_adw(system, 1)

# 時間パラメータ
dt = 1.0      # [fs]
t_end = 200.0  # [fs]

println("\nTime Evolution:")
println("  dt = $dt fs")
println("  t_end = $t_end fs")

println("\nRunning dynamics...")
@time times, pops = evolve(system, Pb0, Pk0, (0.0, t_end), dt)

println("\nResults:")
println("  Initial: p₁=$(pops[1,1]), p₂=$(pops[2,1])")
println("  Final:   p₁=$(pops[1,end]), p₂=$(pops[2,end])")
println("  Total:   $(pops[1,end] + pops[2,end])")

# 中間時点での値も確認
println("\nPopulation at intermediate times:")
for i in [1, 26, 51, 101, 201]
    if i <= length(times)
        println("  t=$(times[i]) fs: p₁=$(round(pops[1,i], digits=6)), p₂=$(round(pops[2,i], digits=6)), total=$(round(pops[1,i]+pops[2,i], digits=6))")
    end
end

# 発散チェック
max_pop = maximum(abs.(pops))
if max_pop > 10
    println("\n⚠️  WARNING: Population diverged! max = $max_pop")
else
    println("\n✓ Population seems stable (max = $max_pop)")
end

println("\n" * "=" ^ 60)
println("Test completed!")
println("=" ^ 60)
