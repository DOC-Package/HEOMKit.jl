# Physical Constants and Units

## Constants

| Constant | Value | Unit | Description |
|----------|-------|------|-------------|
| `kB` | 0.695034800 | cm⁻¹/K | Boltzmann constant |
| `hbar_SI` | 1.054571817×10⁻³⁴ | J·s | Reduced Planck constant |
| `c_light` | 2.99792458×10¹⁰ | cm/s | Speed of light |

## Unit Conversion

HEOMKit uses **inverse centimeters (cm⁻¹)** for energy and **femtoseconds (fs)** for time.

### Conversion Factors

| Factor | Value | Conversion |
|--------|-------|------------|
| `icm2ifs` | ≈ 1.884×10⁻⁴ | cm⁻¹ → fs⁻¹ |
| `ifs2icm` | ≈ 5309 | fs⁻¹ → cm⁻¹ |

### Usage

```julia
# Convert energy: cm⁻¹ → fs⁻¹ (for Hamiltonian)
ε_cm = 100.0  # cm⁻¹
ε_fs = ε_cm * icm2ifs

# Hamiltonian should be in fs⁻¹ units
H = [0 100; 100 0] * icm2ifs
```

## Utility Functions

```julia
# Thermal energy kT at 300 K
kT = thermal_energy(300.0)  # ≈ 208.5 cm⁻¹

# Inverse temperature β = 1/kT at 300 K
β = inverse_temperature(300.0)  # ≈ 0.0048 cm
```

## Pauli Matrices

Standard Pauli matrices are provided for convenience:

```julia
σx = [0 1; 1 0]    # Pauli X
σy = [0 -im; im 0] # Pauli Y  
σz = [1 0; 0 -1]   # Pauli Z
σI = [1 0; 0 1]    # Identity
σp = [0 1; 0 0]    # Raising operator σ⁺
σm = [0 0; 1 0]    # Lowering operator σ⁻
```
