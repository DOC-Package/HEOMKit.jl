"""
    Evolution module for HEOM

HEOM の時間発展を計算する。
"""

# =====================================
# 時間発展積分器
# =====================================

"""
    lsrk4!(P::Matrix{ComplexF64}, dt::Real, system::HEOMSystem; 
           liouvillian::Function=liouville!)

低記憶 4 段 Runge-Kutta 法で 1 ステップ進める（in-place）。

# Arguments
- `P`: ADO 行列 (NL² × nado)
- `dt`: 時間刻み
- `system`: HEOMSystem オブジェクト
- `liouvillian`: Liouville 演算子 (dP, P, system) -> nothing
"""
function lsrk4!(P::Matrix{ComplexF64}, dt::Real, system::HEOMSystem;
                liouvillian::Function=liouville!)
    A = (0.0, -1.0, -1.0, -1.0)
    B = (1.0/3.0, 3.0/4.0, 2.0/3.0, 1.0/4.0)
    
    dP = zeros(ComplexF64, size(P))
    tmp = similar(P)
    
    for (α, β) in zip(A, B)
        liouvillian(tmp, P, system)
        @. dP = α * dP + dt * tmp
        @. P += β * dP
    end
    
    return nothing
end

"""
    lsrk4(P::Matrix{ComplexF64}, dt::Real, system::HEOMSystem;
          liouvillian::Function=liouville!)

低記憶 4 段 Runge-Kutta 法で 1 ステップ進める（新しい配列を返す）。
"""
function lsrk4(P::Matrix{ComplexF64}, dt::Real, system::HEOMSystem;
               liouvillian::Function=liouville!)
    P_new = copy(P)
    lsrk4!(P_new, dt, system; liouvillian=liouvillian)
    return P_new
end


# =====================================
# 時間発展計算
# =====================================

"""
    evolve(system::HEOMSystem, P0::Matrix{ComplexF64}, tspan::Tuple{Real,Real}, dt::Real;
           liouvillian::Function=liouville!, callback=nothing)

HEOM を時間発展させる。

# Arguments
- `system`: HEOMSystem オブジェクト
- `P0`: 初期 ADO (NL² × nado)
- `tspan`: 時間範囲 (t_start, t_end)
- `dt`: 時間刻み
- `liouvillian`: Liouville 演算子 (dP, P, system) -> nothing
- `callback`: 各ステップで呼ばれる関数 (t, P) -> nothing

# Returns
- `times::Vector{Float64}`: 時刻の配列
- `populations::Matrix{Float64}`: 密度行列の対角成分の時間発展

# Example
```julia
times, pops = evolve(system, P0, (0.0, 1000.0), 0.1)

# カスタム Liouville 演算子を使用
times, pops = evolve(system, P0, (0.0, 1000.0), 0.1; liouvillian=my_liouville!)
```
"""
function evolve(system::HEOMSystem, P0::Matrix{ComplexF64}, 
                tspan::Tuple{Real,Real}, dt::Real;
                liouvillian::Function=liouville!, callback=nothing)
    t_start, t_end = tspan
    nsteps = Int(ceil((t_end - t_start) / dt))
    NL = system.NL
    
    # 結果格納
    times = Vector{Float64}(undef, nsteps + 1)
    populations = Matrix{Float64}(undef, NL, nsteps + 1)
    
    # 初期状態
    P = copy(P0)
    t = t_start
    times[1] = t
    
    # ρ₀₀ の対角成分を抽出（vec(ρ) から）
    for i in 1:NL
        idx = (i - 1) * NL + i  # 対角成分のインデックス
        populations[i, 1] = real(P[idx, 1])
    end
    
    if callback !== nothing
        callback(t, P)
    end
    
    # 時間発展
    for step in 1:nsteps
        lsrk4!(P, dt, system; liouvillian=liouvillian)
        t += dt
        times[step + 1] = t
        
        for i in 1:NL
            idx = (i - 1) * NL + i
            populations[i, step + 1] = real(P[idx, 1])
        end
        
        if callback !== nothing
            callback(t, P)
        end
    end
    
    return times, populations
end


# =====================================
# 初期条件
# =====================================

"""
    initial_ado(system::HEOMSystem, rho0::Matrix{ComplexF64})

初期 ADO を作成する。ρ₀ のみ非ゼロ、他の ADO はゼロ。

# Arguments
- `system`: HEOMSystem オブジェクト
- `rho0`: 初期密度行列 (NL × NL)

# Returns
- `P0::Matrix{ComplexF64}`: 初期 ADO (NL² × nado)
"""
function initial_ado(system::HEOMSystem, rho0::Matrix{ComplexF64})
    NL2 = system.NL2
    nado = system.nado
    
    P0 = zeros(ComplexF64, NL2, nado)
    P0[:, 1] = vec(rho0)
    
    return P0
end

"""
    initial_ado(system::HEOMSystem, state::Int=1)

指定した状態 |state⟩⟨state| から始まる初期 ADO を作成する。
"""
function initial_ado(system::HEOMSystem, state::Int=1)
    NL = system.NL
    rho0 = zeros(ComplexF64, NL, NL)
    rho0[state, state] = 1.0
    return initial_ado(system, rho0)
end