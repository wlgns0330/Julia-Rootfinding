

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
- `roundoff::Int = 53`: Bits of precision to solve at. `53` selects `Float64` and the fast
  solver; `<= 24` selects `Float32`, `<= 11` selects `Float16`, and anything above 53
  switches to `BigFloat` at that precision.

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
    dim = length(funcs)
    # polys = Vector{Array{Float64}}()
    polys = Vector{Array{Float64,dim}}(undef, dim)
    errs = fill(0.0,dim)
    # Get an approximation for each function.
    if verbose
        print("Approximation shapes:")
        print(" ")
    end

    # Set precision for Solver
    global type = Float64
    global precision = roundoff

    if precision <= 11
        precision = 11
        type = Float16
    elseif precision <= 24
        precision = 24
        type = Float32
    elseif precision <= 53
        return fast_solve(funcs,a,b; verbose=verbose, returnBoundingBoxes=returnBoundingBoxes, exact=exact, minBoundingIntervalSize=minBoundingIntervalSize,
                          _refine=_refine, _maxDegree=_maxDegree)
    else
        setprecision(precision)
        type = BigFloat
    end

    unitBox = isUnitBox(a, b)
    for i in 1:dim
        if funcs[i] isa MultiPower && unitBox
            polys[i] = multipower_to_cheb(funcs[i].coeff)
            errs[i] = polynomialRoundingError(polys[i], type(2)^-(precision-1))
        elseif funcs[i] isa MultiCheb && unitBox
            polys[i] = funcs[i].coeff
            errs[i] = polynomialRoundingError(polys[i], type(2)^-(precision-1))
        else
            # The approximation's values and coefficients are always Float64 (FFTW's DCT has no other
            # type), so a callable is at best as accurate here as it is in fast_solve. Its sample grid
            # rejects BigFloat bounds, which every sub-box of a BigFloat solve has, so those are
            # rounded to Float64; Float16 and Float32 bounds go through as they are.
            approxA, approxB = type == BigFloat ? (Float64.(a), Float64.(b)) : (a, b)
            polys[i], errs[i] = chebApproximate(funcs[i],approxA,approxB; maxDegree=_maxDegree)
            # The solver below only allows for rounding in `type`, so the approximation's own Float64
            # rounding has to be in its error. Without it the boxes of a BigFloat solve came out narrower
            # than that rounding, missing the root, and two roots 1e-9 apart were both lost.
            errs[i] = max(errs[i], polynomialRoundingError(polys[i], eps(Float64)))
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

    polys = [type.(arr) for arr in polys]
    errs = type.(errs)
    a = type.(a)
    b = type.(b)

    minBoundingIntervalSize = type(minBoundingIntervalSize)
    
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
        #Subdivide the interval and resolve to get better resolution across different parts of the interval
        yroots, boundingBoxes = [], []
        for val in Iterators.product(Iterators.repeated(([false,true]), length(a))...)
            #Split almost in half
            #TODO: Do we need to combine bounding boxes in this step of the recursion as well?
            #      For now it seems safe enough to assume we won't have any roots on the midpoints.
            val = reverse(val)
            midPoint = subdivisionPoint(a, b, type(0.51234912839471234))
            newA = ifelse.(val,midPoint,a)
            newB = ifelse.(val,b,midPoint)
            #Solve recursively
            if verbose
                print("Re-solving on: ")
                print(newA)
                print(" ")
                println(newB)
            end
            roots, boxes = solve(funcs, newA, newB; verbose=verbose, returnBoundingBoxes=true, exact=exact, minBoundingIntervalSize = minBoundingIntervalSize, roundoff=roundoff,
                                 _refine=_refine, _maxDegree=_maxDegree)
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

    #TODO: Handle if we have duplicate roots or extra roots at the top level. Easiest if we actually return the bounding boxes!
    #Maybe return the bounding boxes in the recursive steps?
    
    #If any of the bounding boxes is too large, re-solve that box.
    # Each entry is (roots, boxes, isWide); see _refineWideBoxes.
    entries = []
    scale = max.(abs.(a), abs.(b), 1)
    # A simple root's box ends up a few macheps wide, so a box is only wide once it is well past that.
    # The approximation is built in Float64, so no box is held to more than a Float64 solve is.
    wideSize = max(type(REFINE_BOX_SIZE), eps(type)^(5//8))
    for box in boundingBoxes
        #Get the relative max size in each dimension. If a or b > 1 in magnitude, minBoundingIntervalSize is a relative number.
        #If they are < 1 in magnitude, it is an absolute number.
        newBox = transformPoints(box.finalInterval',a,b)
        newA, newB = newBox[:,1],newBox[:,2]

        relMaxSize = minBoundingIntervalSize .* maximum(hcat(abs.(a),abs.(b),fill(type(1),length(a))),dims=2)
        if all(newB - newA .> relMaxSize)
            #Re-solve this box
            if verbose
                print("Re-solving on: ")
                print(newA)
                print(" ")
                println(newB)
            end
            roots, boxes = solve(funcs, newA, newB; verbose=verbose, returnBoundingBoxes=true, exact=exact, minBoundingIntervalSize = minBoundingIntervalSize, roundoff=roundoff,
                                 _refine=_refine, _maxDegree=_maxDegree)
            if length(roots) > 0
                push!(entries, (collect(roots), collect(boxes), false))
            end
        else
            push!(entries, finalBoxEntry(box, a, b, transformPoints, getFinalPoint, scale, wideSize))
        end
    end

    if _refine && any(entry[3] for entry in entries) && !allPolynomial(funcs)
        entries = _refineWideBoxes(solve, funcs, a, b, entries; verbose=verbose, exact=exact,
                                   minBoundingIntervalSize=minBoundingIntervalSize, roundoff=roundoff)
    end
    finalRoots = Any[r for entry in entries for r in entry[1]]
    finalBoxes = Any[bx for entry in entries for bx in entry[2]]
    # Find and return the roots (and, optionally, the bounding boxes)
    if returnBoundingBoxes
        return finalRoots, finalBoxes
    else
        return finalRoots
    end
end
