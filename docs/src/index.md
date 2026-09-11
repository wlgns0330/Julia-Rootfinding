# YRoots.jl

*A fast-working package for finding the roots of multivariate systems of equations.*

## How YRoots Works

YRoots harnesses the properties of Chebyshev polynomial approximation to quickly and
precisely find and return the roots of various systems of functions.

Given a list of smooth, continuous functions and a compact search interval, YRoots
generates an accurate approximation for each function on the interval and recursively
uses numerical methods to zero in on any roots contained in the interval.

## Getting Started

```julia
using YRoots

# Roots of x + y = 0.3 and x - y = 0.1 on [-1,1]^2
roots = solve([(x, y) -> x + y - 0.3,
               (x, y) -> x - y - 0.1], [-1.0, -1.0], [1.0, 1.0])
```

See [`solve`](@ref) for the full parameter reference, and
[Polynomials](@ref) for the `MultiCheb` and `MultiPower` helper types.

!!! note "First call is slow"
    YRoots specialises the solver on the dimension of the system. The package
    precompiles dimensions 1 through 5 at install time, so the first solve of a
    supported dimension returns promptly; a higher dimension pays a one-off
    compilation cost on its first call.

!!! warning "Requirements on the input"
    `solve` is only guaranteed to work well when each function is continuous and
    smooth on the search interval and every root in it is simple. A function that is
    not smooth, or an interval containing infinitely many roots, may send the solver
    into deep subdivision.
