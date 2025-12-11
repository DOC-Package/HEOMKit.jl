# HEOM System

## Theory

The Hierarchical Equations of Motion (HEOM) describes the dynamics of a reduced density matrix coupled to a bosonic bath:

```math
\frac{d\rho_\mathbf{n}}{dt} = -i[H, \rho_\mathbf{n}] - \sum_k n_k \gamma_k \rho_\mathbf{n} 
- i \sum_k [V, \Phi_k \rho_{\mathbf{n}+\mathbf{e}_k}]
- i \sum_k (V \rho_{\mathbf{n}-\mathbf{e}_k} \Theta_k^L - \rho_{\mathbf{n}-\mathbf{e}_k} V^\dagger \Theta_k^R)
```

where:
- ``\rho_\mathbf{n}`` is the auxiliary density operator (ADO) with index ``\mathbf{n} = (n_1, n_2, \ldots)``
- ``\Phi_k = \sqrt{(n_k+1)|c_k|}`` are forward connection coefficients
- ``\Theta_k`` are backward connection coefficients

## Building a HEOM System

```julia
using KaisouEOM

# System Hamiltonian
H = [0 100; 100 0] * icm2ifs

# Bath parameters
γ = [0.01 + 0.001im]
c = [0.001]
V = [1 0; 0 -1]

bath = Bath(γ, c, V)
noise = Noise(bath)

# Build HEOM system with hierarchy depth 5
system = HEOMSystem(H, noise, 5)
```

## Hierarchy Truncation

Two methods are available:

### Depth-based (`:depth`)

Include all ADOs with ``\sum_k n_k \leq N_{\text{max}}``:

```julia
system = HEOMSystem(H, noise, 5; hierarchy=:depth)
```

### Width-based (`:width`)

Filter ADOs based on importance criteria:

```julia
system = HEOMSystem(H, noise, 5; hierarchy=:width)
```

## Liouville Operator

Apply the HEOM Liouvillian:

```julia
# In-place version
dP = similar(P)
liouville!(dP, P, system)

# Allocating version
dP = liouville(P, system)
```
