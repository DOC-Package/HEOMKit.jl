```@meta
CurrentModule = HEOMKit
```

# HEOMKit.jl

HEOMKit.jl is a Julia package for Hierarchical Equations of Motion (HEOM) calculations in open quantum systems.

## Features

- Efficient HEOM implementation with sparse matrices
- Support for multiple baths (Drude-Lorentz, Brownian, custom)
- Flexible hierarchy truncation (depth-based or width-based)
- Low-storage Runge-Kutta time integrator
- File output support for long simulations

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/htkhsh/HEOMKit.jl")
```

## Quick Start

```julia
using HEOMKit

# Define system Hamiltonian (2-level system)
ε = 0.0      # Energy gap [cm⁻¹]
Δ = 100.0    # Tunneling [cm⁻¹]
H = [ε Δ; Δ -ε] * icm2ifs

# Define bath parameters
γ = [0.01 + 0.001im]   # Exponential decay rate
c = [0.001 + 0.0001im] # Expansion coefficient
V = [1.0 0.0; 0.0 -1.0] # System-bath coupling

# Create bath and noise
bath = BathExp(γ, c, V)
noise = NoiseExp(bath)

# Build HEOM system
system = HEOMSystem(H, noise, 5)

# Time evolution
P0 = initial_ado(system, 1)  # Start from |1⟩⟨1|
times, pops = evolve(system, P0, (0.0, 500.0), 1.0)
```

## Manual

```@contents
Pages = [
    "manual/bath.md",
    "manual/heom.md",
    "manual/evolution.md",
    "manual/performance.md",
    "manual/constants.md",
]
Depth = 2
```

## API Reference

```@contents
Pages = ["api.md"]
Depth = 2
```
