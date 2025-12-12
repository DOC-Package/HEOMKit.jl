module KaisouEOM

using LinearAlgebra
using SparseArrays

# const.jl: Physical constants and unit conversion
export kB, hbar_SI, c_light
export icm2ifs, ifs2icm
export thermal_energy, inverse_temperature

# noise.jl: Bath and Noise structures
export Bath, Noise
export drude_bath, brownian_bath
export compute_heom_params

# hierarchy.jl: Hierarchy index construction
export hierarchy_index_depth, hierarchy_index_width
export build_hseom_index_maps

# matrix.jl: Liouville space operators
export σx, σy, σz, σI, σp, σm
export matx, mato, matl, matr
export HEOMMatrices

# heom.jl: HEOM structures and Liouville operator
export HEOMOperators, HEOMSystem
export liouville!, liouville

# hseom.jl: HSEOM structures and Liouville operators (wavefunction-based)
export HSEOMOperators, HSEOMSystem
export liouville_ket!, liouville_ket
export liouville_bra!, liouville_bra
export initial_adw
export build_tridiagonal_D

# evolution.jl: Time evolution
export lsrk4!, lsrk4
export evolve
export initial_ado

# Include files
include("const.jl")
include("noise.jl")
include("hierarchy.jl")
include("matrix.jl")
include("heom.jl")
include("hseom.jl")
include("evolution.jl")

end
