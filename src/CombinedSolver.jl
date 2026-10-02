

"""
Bound the error of a polynomial given exactly as coefficients: rounding in those coefficients.

Each coefficient is off by at most `macheps` times its own size, so the polynomial is off by at
most `macheps` times the sum of their sizes anywhere on [-1,1]^n. A fixed `macheps` instead is only
right when the coefficients are of order 1: from 1e-16 down it exceeds every coefficient, and the
solver overflowed the stack.

The identically zero polynomial keeps the fixed `macheps`. Its relative error is 0, which makes the
solver report no roots for a polynomial that vanishes everywhere.
"""
function polynomialRoundingError(coeff, macheps)
    absSum = sum(abs, coeff)
    return absSum > 0 ? macheps*absSum : macheps
end

"""
Check the arguments of [`solve`](@ref) and put the bounds in the form it works with, so bad input fails
up front with an `ArgumentError` naming the problem instead of deep inside the solver, or not at all.

Returns `(funcs, a, b)`, with `a` and `b` as new floating-point vectors of length `length(funcs)`; a
single number for a bound is used in every dimension. The arguments themselves are left unchanged.
"""
function validateSolveInput(funcs, a, b)
    (funcs isa AbstractVector || funcs isa Tuple) ||
        throw(ArgumentError("funcs must be a vector of functions, one per dimension; got a $(typeof(funcs))"))
    dim = length(funcs)
    dim > 0 || throw(ArgumentError("funcs is empty; give at least one function"))
    bound(v, name) = if v isa Real
        fill(float(v), dim)
    elseif v isa AbstractVector{<:Real}
        length(v) == dim || throw(ArgumentError(
            "$name has $(length(v)) entries but there are $dim functions; give one bound per dimension"))
        float.(collect(v))
    else
        throw(ArgumentError("$name must be a number or a vector of numbers; got a $(typeof(v))"))
    end
    a = bound(a, "a")
    b = bound(b, "b")
    all(isfinite, a) && all(isfinite, b) ||
        throw(ArgumentError("the bounds must be finite; got a = $a, b = $b"))
    for i in 1:dim
        a[i] < b[i] || throw(ArgumentError(
            "the lower bound must be below the upper bound in every dimension; in dimension $i, a = $(a[i]) and b = $(b[i])"))
    end
    for (i, f) in enumerate(funcs)
        if f isa MultiPower || f isa MultiCheb
            f.dim == dim || throw(ArgumentError(
                "function $i is a polynomial in $(f.dim) variable(s), but the system has $dim functions"))
        elseif !applicable(f, a...)
            throw(ArgumentError(f isa Function ?
                "function $i cannot be called with $dim arguments, one per dimension" :
                "function $i is a $(typeof(f)), which is not callable; give a function, MultiPower or MultiCheb"))
        end
    end
    return funcs, a, b
end

"""
The number type `solve` runs in for `roundoff` bits of precision: Float16 for 11 or fewer, Float32 for 24
or fewer, Float64 for 53 or fewer, and BigFloat (at `roundoff` bits) above that.
"""
solveType(roundoff) = roundoff <= 11 ? Float16 : roundoff <= 24 ? Float32 : roundoff <= 53 ? Float64 : BigFloat

"""
    solve(funcs, a, b; verbose=false, returnBoundingBoxes=false, exact=false,
          minBoundingIntervalSize=1e-5, roundoff=53)

Find the roots of a system of functions on the search interval `[a, b]`.

Generates a Chebyshev approximation for each function on the given interval, then uses
properties of those approximations to shrink the search interval. When the information in
the approximation is insufficient to shrink it further, the interval is subdivided and the
search recurses until it zeros in on each root. One point -- and optionally a bounding box
-- is returned per root found.

# Arguments
- `funcs`: Vector of functions to solve simultaneously. Each element is either a callable
  taking one argument per dimension, or a [`MultiCheb`](@ref) / [`MultiPower`](@ref)
  polynomial.
- `a`: Vector holding the lower bound of the search interval in each dimension, in
  dimension order, or a single number to use in every dimension.
- `b`: The upper bound, likewise. It must be above `a` in every dimension, and both must
  be finite; `solve` throws an `ArgumentError` otherwise, and when a function cannot be
  called with one argument per dimension or a polynomial's dimension is not the number of
  functions.

# Keyword arguments
- `verbose::Bool = false`: Print progress of the approximation and rootfinding to the
  terminal. Useful for systems that take a long time to solve.
- `returnBoundingBoxes::Bool = false`: Also return a bounding box for each root.
- `exact::Bool = false`: Run the transformations on the approximation in higher precision,
  minimising error at some cost in speed.
- `minBoundingIntervalSize::Real = 1e-5`: If a root is found whose bounding interval is
  larger than this in every dimension, the system is solved again on the smaller interval.
  Smaller values give more accurate roots but increase solve time, and can cause trouble
  if the functions cannot be evaluated accurately at points close together. The value is
  absolute while the interval in question lies within `[-1, 1]`, and relative otherwise:
  for an interval with an endpoint of magnitude greater than 1, it is multiplied by that
  magnitude in that dimension.
- `roundoff::Int = 53`: Bits of precision to solve at. `53` (the default, and anything from 25
  to 53) solves in `Float64`, `<= 24` in `Float32`, `<= 11` in `Float16`, and anything above 53 in
  `BigFloat` at that precision. Every precision runs the same solver. A callable is always
  approximated in `Float64`, so a higher precision can only improve the accuracy of polynomial
  input ([`MultiPower`](@ref) / [`MultiCheb`](@ref)).

# Returns
A vector of roots, each a point in the search interval. With `returnBoundingBoxes = true`,
returns `(roots, boundingBoxes)` instead, where each box is a `2 x dim` array whose first
row holds the lower bound in each dimension and whose second row holds the upper bound.

# Examples
```julia
julia> using YRoots

julia> solve([(x, y) -> x + y - 0.3, (x, y) -> x - y - 0.1], [-1.0, -1.0], [1.0, 1.0])
1-element Vector{Any}:
 [0.2, 0.09999999999999994]

julia> M1 = MultiPower([0 3 0 2; 1.5 0 7 0; 0 0 4 -2; 0 0 0 1]);

julia> M2 = MultiCheb([0.02 0.31; -0.43 0.19; 0.06 0]);

julia> solve([M1, M2], [-5.0, -5.0], [5.0, 5.0]);
```

!!! note "First solve of a given dimension"
    The solver is specialised on the dimension of the system. Dimensions 1 through 5 are
    precompiled when the package is installed, so their first solve returns promptly; a
    higher dimension pays a one-off compilation cost of several seconds on its first call.

!!! warning "Requirements on the input"
    `solve` is only guaranteed to work well when every function is continuous and smooth
    on the search interval and every root in it is simple. A function that is not smooth,
    or an interval containing infinitely many roots, may drive the solver into deep
    subdivision; it will give up at `maxLevel` and report a wide bounding box.
"""

function solve(funcs,a,b; verbose = false, returnBoundingBoxes = false, exact=false, minBoundingIntervalSize=1e-5, roundoff=53,
               _refine=true, _maxDegree=nothing)
    funcs, a, b = validateSolveInput(funcs, a, b)
    # Every precision runs the same solver, generic over its number type: the one that used to be
    # Float64-only. It solves sub-boxes by calling itself, so the input above is checked once.
    T = solveType(roundoff)
    run() = fast_solve(funcs, a, b; verbose=verbose, returnBoundingBoxes=returnBoundingBoxes, exact=exact,
                       minBoundingIntervalSize=minBoundingIntervalSize, _refine=_refine, _maxDegree=_maxDegree,
                       numtype=T)
    # BigFloat's precision is a global setting; it is set only for this solve, not left changed for the caller.
    return T === BigFloat ? setprecision(run, BigFloat, roundoff) : run()
end
