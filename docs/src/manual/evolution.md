# Time Evolution

## Initial State

Create initial ADO from a density matrix or pure state:

```julia
# From density matrix
rho0 = [1.0 0.0; 0.0 0.0]
P0 = initial_ado(system, ComplexF64.(rho0))

# From pure state |n⟩⟨n|
P0 = initial_ado(system, 1)  # |1⟩⟨1|
P0 = initial_ado(system, 2)  # |2⟩⟨2|
```

## Time Evolution

The `evolve` function performs time evolution and returns populations:

```julia
# Basic usage
times, populations = evolve(system, P0, (0.0, 500.0), 1.0)

# With file output
times, populations = evolve(system, P0, (0.0, 500.0), 1.0;
                           savefile="output.dat", save_interval=100)

# With callback function
callback(t, P) = println("t = $t")
times, populations = evolve(system, P0, (0.0, 500.0), 1.0; callback=callback)
```

## Single Step Integration

For custom time evolution, use the LSRK4 integrator directly:

```julia
# In-place (modifies P)
lsrk4!(P, dt, system)

# Allocating (returns new array)
P_new = lsrk4(P, dt, system)
```

## Custom Liouvillian

You can provide a custom Liouvillian function:

```julia
function my_liouvillian!(dP, P, system)
    liouville!(dP, P, system)
    # Add custom terms...
end

times, populations = evolve(system, P0, tspan, dt; liouvillian=my_liouvillian!)
```
