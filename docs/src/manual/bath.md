# Bath and Noise

## Bath Correlation Function

In HEOM, the bath correlation function is expanded as a sum of exponentials:

```math
C(t) = \sum_k c_k e^{-\gamma_k t}
```

where ``\gamma_k`` are the exponential decay rates and ``c_k`` are the expansion coefficients.

## Bath Structure

The `BathExp` structure holds the expansion parameters for a single bath:

```julia
# Direct construction with exponential parameters
γ = [0.01 + 0.001im, 0.02 - 0.001im]
c = [0.001 + 0.0001im, 0.0005 - 0.0001im]
V = [1.0 0.0; 0.0 -1.0]  # System-bath coupling operator

bath = BathExp(γ, c, V)
```

`BathExp` stores the original exponential expansion. Missing complex-conjugate modes are added later when you build `NoiseExp`.

## Noise Structure

The `NoiseExp` structure combines multiple baths:

```julia
# Single bath
noise = NoiseExp(bath)

# Multiple baths
bath1 = BathExp(γ1, c1, V1)
bath2 = BathExp(γ2, c2, V2)
noise = NoiseExp([bath1, bath2])
```

## Spectral Densities

Common spectral densities can be used with ESPRIT fitting (requires ExpFit.jl):

### Drude-Lorentz (Overdamped)

```math
J(\omega) = \frac{2\lambda\gamma\omega}{\omega^2 + \gamma^2}
```

### Brownian (Underdamped)

```math
J(\omega) = \frac{4\lambda\gamma\Omega^2\omega}{(\omega^2 - \Omega^2)^2 + 4\gamma^2\omega^2}
```
