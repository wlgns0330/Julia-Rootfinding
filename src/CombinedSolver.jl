

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
    warnDuplicateRoots(dupSets)

Warn once about every set of roots that could not be told apart from each other.

`dupSets` holds one vector of roots per bounding box that reports more than one root, in the original
coordinates. Nothing is printed when it is empty.
"""
function warnDuplicateRoots(dupSets)
    isempty(dupSets) && return
    formatRoot(root) = "(" * join((@sprintf("%.16g", x) for x in root), ", ") * ")"
    lines = ["  Set $i: " * join((formatRoot(root) for root in roots), ", ") for (i, roots) in enumerate(dupSets)]
    @warn "The roots in each of the following sets might be duplicates of each other:\n" *
          join(lines, "\n") *
          "\nWe suggest you check the rank of the Jacobian at these points."
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
The `minBoundingIntervalSize` a solve in floating-point type `T` uses when none is given.

`Float64` and `BigFloat` keep the long-standing `1e-5`. A callable is approximated in `Float64` whatever
the type, so a `BigFloat` solve of one cannot tighten its boxes past a `Float64` solve.

`Float32` uses `eps(Float32)^(1/3)`, about `4.9e-3`, the same cube root of machine epsilon that `1e-5` is
for `Float64`. Its boxes come out `1e-6` to `2e-4` wide, so under `1e-5` nearly every one was solved again,
which made a Float32 solve up to 8x slower than a Float64 one for accuracy it was never asked for.

`Float16` uses `8*eps(Float16)`, about `7.8e-3`: the floor the re-solve already applies, since a box only a
few rounding steps wide cannot be shrunk. Its boxes are only a few steps wide to begin with, and the
re-solve is what splits or discards the wider ones; at `eps(Float16)^(1/3)` (about `0.1`) or even `0.03`
they were kept, and a system with 5 roots came back with 7.
"""
defaultMinBoundingIntervalSize(T) = T === Float16 ? 8*eps(Float16) : T === Float32 ? eps(Float32)^(1/3) : 1e-5

"""
    solve(funcs, a, b; verbose=false, returnBoundingBoxes=false, exact=false,
          minBoundingIntervalSize=nothing, roundoff=53)

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
- `minBoundingIntervalSize::Real`: If a root is found whose bounding interval is
  larger than this in every dimension, the system is solved again on the smaller interval.
  Smaller values give more accurate roots but increase solve time, and can cause trouble
  if the functions cannot be evaluated accurately at points close together. The value is
  absolute while the interval in question lies within `[-1, 1]`, and relative otherwise:
  for an interval with an endpoint of magnitude greater than 1, it is multiplied by that
  magnitude in that dimension. Defaults to a value for the floating-point type `roundoff`
  selects: `1e-5` for `Float64` and `BigFloat`, `eps(Float32)^(1/3)` (about `4.9e-3`) for
  `Float32` and `8*eps(Float16)` (about `7.8e-3`) for `Float16`.
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

function solve(funcs,a,b; verbose = false, returnBoundingBoxes = false, exact=false, minBoundingIntervalSize=nothing, roundoff=53)
    funcs, a, b = validateSolveInput(funcs, a, b)
    # Every precision runs the same solver, generic over its number type. It solves sub-boxes by
    # calling itself, so the input above is checked once.
    T = solveType(roundoff)
    minBoundingIntervalSize = something(minBoundingIntervalSize, defaultMinBoundingIntervalSize(T))
    # The solves it makes add their sets of possible duplicate roots here, so this call warns about all of them once.
    dupSets = Any[]
    run() = solveSystem(funcs, a, b; verbose=verbose, returnBoundingBoxes=returnBoundingBoxes, exact=exact,
                       minBoundingIntervalSize=minBoundingIntervalSize, numtype=T, dupSets=dupSets)
    # BigFloat's precision is a global setting; it is set only for this solve, not left changed for the caller.
    result = T === BigFloat ? setprecision(run, BigFloat, roundoff) : run()
    warnDuplicateRoots(dupSets)
    return result
end

"""
The point where a search interval `[a, b]` that could not be shrunk is split, `frac` of the way from
`a` to `b` in each dimension; `frac` is just off one half, so a root on the exact midpoint does not land
on the split. Computed from the width, so it stays inside the interval wherever the interval is: scaling
`a + b` instead gave a point outside it for an interval away from the origin: -101.96 for `[-100, -99]`.
"""
subdivisionPoint(a, b, frac) = a .+ (b .- a) .* frac

"""
Whether `[a, b]` is the box [-1, 1]^n that a polynomial's coefficients are written on. On any other box,
including the sub-boxes a solve recurses on, MultiPower and MultiCheb coefficients are first carried
onto it by the linear change of variables x = (b - a)/2 * t + (b + a)/2, as the Python solver does.
"""
isUnitBox(a, b) = all(a .== -1) && all(b .== 1)

"""
Whether every function is a polynomial given as coefficients.

A polynomial is carried onto a smaller box by transforming its coefficients, and the error of the
transformed coefficients stays that of the original ones however small the box, while a callable is
approximated afresh on it, with an error that shrinks with the function's size there.
"""
allPolynomial(funcs) = all(f isa MultiPower || f isa MultiCheb for f in funcs)

"""
Whether the Chebyshev polynomial `coeffs`, with error bound `err`, is swamped by its error on its region:
whether the error is at least the most the polynomial can vary there (the sum of its non-constant
coefficients), so that it cannot be told apart from a constant anywhere on the region.

Carried onto a smaller region, a polynomial varies less while its error bound stays the same, so a
swamped polynomial stays swamped on every part of the region, and cutting the region up cannot locate
its roots. This is what happens to polynomial input in Float32 and Float16 near its zero curves.
"""
swampedByError(coeffs, err) = err >= sum(abs, coeffs) - abs(coeffs[1])

"""
The roots and boxes one final box of the subdivision solve contributes, in the original coordinates,
as an entry `(roots, boxes, isDupSet)`.

The box is repeated once per root, so the roots and boxes a solve returns stay paired even when the
final step keeps several possible duplicates in one box. `isDupSet` says whether the box reports more
than one root, which could not be told apart.
"""
function finalBoxEntry(box, a, b)
    transformedBox = transformPoints(box.finalInterval', a, b)'
    boxRoots = Any[]
    if length(box.possibleDuplicateRoots) > 0
        for dup in box.possibleDuplicateRoots
            push!(boxRoots, transformPoints(dup, a, b))
        end
    else
        push!(boxRoots, transformPoints(getFinalPoint(box), a, b))
    end
    return (boxRoots, Any[transformedBox for _ in boxRoots], length(boxRoots) > 1)
end

"""
The error bound for a Float64 approximation `coeffs` (whose own error is `approxError`) once it is
converted to the solve's type T.

The solver allows only for rounding in T. In Float64 that already covers the approximation, so the bound
is unchanged. Below Float64 the conversion to T rounds each coefficient again; above it the Float64
rounding of the approximation is larger than anything the solver allows for. Either way the coarser of
the two is added to the bound, rounded up so converting the bound to T cannot shrink it. Without it,
Float32 returned no roots for a pair 1e-6 apart and an 80-bit BigFloat solve for one 1e-9 apart.
"""
function convertedApproxError(::Type{T}, coeffs, approxError) where {T}
    T === Float64 && return approxError
    bound = approxError + polynomialRoundingError(coeffs, max(eps(Float64), Float64(eps(T))))
    return nextfloat(T(bound))
end

"""
    solveSystem(funcs, a, b; verbose, returnBoundingBoxes, exact, minBoundingIntervalSize,
                numtype=Float64, dupSets=Any[])

The solver behind [`solve`](@ref), in the number type `numtype`, on input `solve` has already checked.
It solves sub-boxes by calling itself, so that check is not repeated on each of them. See `solve` for
the arguments. Each set of roots a bounding box could not separate is added to `dupSets`, in the
original coordinates, for `solve` to warn about once.
"""
function solveSystem(funcs,a,b; verbose, returnBoundingBoxes, exact, minBoundingIntervalSize,
                    numtype::Type{<:AbstractFloat}=Float64, dupSets=Any[])
    T = numtype
    dim = length(funcs)
    polys = Vector{Array{T,dim}}(undef, dim)
    errs = zeros(T, dim)
    # Get an approximation for each function.
    if verbose
        print("Approximation shapes:")
        print(" ")
    end

    unitBox = isUnitBox(a, b)
    for i in 1:dim
        if funcs[i] isa MultiPower || funcs[i] isa MultiCheb
            # The coefficients are converted to T first, so the basis conversion and the change of
            # variables onto the box are both done in the solve's precision.
            coeff = T.(funcs[i].coeff)
            polys[i] = funcs[i] isa MultiPower ? multipower_to_cheb(coeff) : coeff
            errs[i] = polynomialRoundingError(polys[i], eps(T))
            if !unitBox
                polys[i], errs[i] = transformCheb(polys[i], T.((b .- a) ./ 2), T.((b .+ a) ./ 2), errs[i], exact)
            end
        else
            # Always approximated in Float64 (FFTW's DCT has no other type), then converted.
            approx, approxError = chebApproximate(funcs[i], Float64.(a), Float64.(b))
            polys[i] = T.(approx)
            errs[i] = convertedApproxError(T, approx, approxError)
        end
        if verbose
            print(i)
            print(": ")
            print(reverse(size(polys[i])))
            if i != dim
                print(" ")
            else
                print("\n")
            end
        end
    end

    if verbose
        print("Searching on interval ")
        println([[a[i],b[i]] for i in 1:dim])
    end

    #Solve the Chebyshev polynomial system
    yroots, boundingBoxes = solveChebyshevSubdivision(polys,errs;verbose=verbose,returnBoundingBoxes=true,exact=exact,
                constant_check=true, low_dim_quadratic_check=true, all_dim_quadratic_check=false)

    #If the bounding box is the entire interval, subdivide it!
    usingSubdivision = all(b-a .> minBoundingIntervalSize)
    if length(boundingBoxes) == 1 && all(finalDimSize(boundingBoxes[1]) .== 2) && usingSubdivision
        # A polynomial swamped by its error stays swamped on every part of the interval (see
        # swampedByError), so subdividing would only tile it down to minBoundingIntervalSize and return
        # every tile as a root: in Float32, 3,830 roots for 4 on one system and a run that never ended on
        # another. Return the interval as one box instead.
        if allPolynomial(funcs) && any(swampedByError(polys[i], errs[i]) for i in 1:dim)
            @warn "Part of the interval cannot be resolved at this precision; it is returned as one bounding box." maxlog=1
            box = permutedims(hcat(a, b))
            root = (a .+ b) ./ 2
            return returnBoundingBoxes ? (Any[root], Any[box]) : Any[root]
        end
        #Subdivide the interval and resolve to get better resolution across different parts of the interval
        yroots, boundingBoxes = [], []
        for val in Iterators.product(Iterators.repeated(([false,true]), length(a))...)
            #Split almost in half
            #TODO: Do we need to combine bounding boxes in this step of the recursion as well?
            #      For now it seems safe enough to assume we won't have any roots on the midpoints.
            val = reverse(val)
            midPoint = subdivisionPoint(a, b, 0.51234912839471234)
            newA = ifelse.(val,midPoint,a)
            newB = ifelse.(val,b,midPoint)
            #Solve recursively
            if verbose
                print("Re-solving on: ")
                print(newA)
                print(" ")
                println(newB)
            end
            roots, boxes = solveSystem(funcs, newA, newB; verbose=verbose, returnBoundingBoxes=true, exact=exact, minBoundingIntervalSize = minBoundingIntervalSize,
                                      numtype=T, dupSets=dupSets)
            if length(roots) != 0
                append!(boundingBoxes,boxes)
                append!(yroots,roots)
            end
        end
        if returnBoundingBoxes
            return yroots, boundingBoxes
        else
            return yroots
        end
    end

    #TODO: Handle if we have extra roots at the top level.
    
    #If any of the bounding boxes is too large, re-solve that box.
    # Each entry is (roots, boxes, isDupSet); see finalBoxEntry.
    entries = []
    for box in boundingBoxes
        #Get the relative max size in each dimension. If a or b > 1 in magnitude, minBoundingIntervalSize is a relative number.
        #If they are < 1 in magnitude, it is an absolute number.
        newBox = transformPoints(box.finalInterval',a,b)
        newA, newB = newBox[:,1],newBox[:,2]

        # A box only a few rounding steps of T wide cannot be shrunk by solving it again, and in Float16 the
        # default 1e-5 is below one step (2.4e-4 near 0.3): solving such a box again found nothing and its
        # root was dropped. The floor of 8 steps is below 1e-5 for Float32 and above, so it changes nothing there.
        relMaxSize = max(minBoundingIntervalSize, 8*eps(T)) .* maximum(hcat(abs.(a),abs.(b),fill(1,length(a))),dims=2)
        if all(newB - newA .> relMaxSize)
            #Re-solve this box
            if verbose
                print("Re-solving on: ")
                print(newA)
                print(" ")
                println(newB)
            end
            roots, boxes = solveSystem(funcs, newA, newB; verbose=verbose, returnBoundingBoxes=true, exact=exact, minBoundingIntervalSize = minBoundingIntervalSize,
                                      numtype=T, dupSets=dupSets)
            if length(roots) > 0
                push!(entries, (collect(roots), collect(boxes), false))
            end
        else
            push!(entries, finalBoxEntry(box, a, b))
        end
    end

    # A box that could not separate its roots reports more than one.
    append!(dupSets, [entry[1] for entry in entries if entry[3]])
    finalRoots = Any[r for entry in entries for r in entry[1]]
    finalBoxes = Any[bx for entry in entries for bx in entry[2]]
    # Find and return the roots (and, optionally, the bounding boxes)
    if returnBoundingBoxes
        return finalRoots, finalBoxes
    else
        return finalRoots
    end
end
