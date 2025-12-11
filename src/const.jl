"""Physical constants for HEOM calculations

Define physical constants and unit conversion factors.
"""

# =====================================
# Fundamental physical constants
# =====================================

"""Boltzmann constant (cm⁻¹/K), CODATA 2018"""
const kB = 0.695034800

"""Reduced Planck constant (J·s)"""
const hbar_SI = 1.054571817e-34

"""Speed of light (cm/s)"""
const c_light = 2.99792458e10


# =====================================
# Unit conversion factors
# =====================================

"""Conversion factor: cm⁻¹ → fs⁻¹ (≈ 1.884×10⁻⁴)"""
const icm2ifs = 2π * c_light * 1e-15

"""Conversion factor: fs⁻¹ → cm⁻¹"""
const ifs2icm = 1.0 / icm2ifs


# =====================================
# Utility functions
# =====================================

"""    thermal_energy(T) → kT in cm⁻¹"""
thermal_energy(T::Real) = kB * T

"""    inverse_temperature(T) → β = 1/kT in cm"""
inverse_temperature(T::Real) = 1.0 / (kB * T)
