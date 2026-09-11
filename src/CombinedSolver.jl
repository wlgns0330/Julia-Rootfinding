include("ChebyshevApproximator.jl")
include("ChebyshevSubdivisionSolver.jl")
include("StructsWithTheirFunctions/Polynomial.jl")
include("FastSolve/FastCombinedSolver.jl")


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
  dimension order.
- `b`: Vector holding the upper bound, likewise.

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
function solve(funcs,a,b; verbose = false, returnBoundingBoxes = false, exact=false, minBoundingIntervalSize=1e-5, roundoff=53)
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
        return fast_solve(funcs,a,b; verbose=verbose, returnBoundingBoxes=returnBoundingBoxes, exact=exact, minBoundingIntervalSize=minBoundingIntervalSize)
    else
        setprecision(precision)
        type = BigFloat
    end

    for i in 1:dim
        if typeof(funcs[i]) == MultiPower
            polys[i] = multipower_to_cheb(funcs[i].coeff)
            errs[i] = type(2)^-(precision-1)
        elseif typeof(funcs[i]) == MultiCheb
            polys[i] = funcs[i].coeff
            errs[i] = type(2)^-(precision-1)
        else
            polys[i], errs[i] = chebApproximate(funcs[i],a,b)
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
            midPoint = (a + b) .* type(0.51234912839471234)
            newA = ifelse.(val,midPoint,a)
            newB = ifelse.(val,b,midPoint)
            #Solve recursively
            if verbose
                print("Re-solving on: ")
                print(newA)
                print(" ")
                println(newB)
            end
            roots, boxes = solve(funcs, newA, newB; verbose=verbose, returnBoundingBoxes=true, exact=exact, minBoundingIntervalSize = minBoundingIntervalSize, roundoff=roundoff)
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
    finalBoxes = []
    finalRoots = []
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
            roots, boxes = solve(funcs, newA, newB; verbose=verbose, returnBoundingBoxes=true, exact=exact, minBoundingIntervalSize = minBoundingIntervalSize, roundoff=roundoff)
            if length(roots) > 0
                append!(finalRoots,roots)
                append!(finalBoxes,boxes)
            end
        else
            #Transform back
            push!(finalBoxes,transformPoints(box.finalInterval',a,b)')
            #Get the roots from this box
            if length(box.possibleDuplicateRoots) > 0
                for dup in box.possibleDuplicateRoots
                    push!(finalRoots,transformPoints(dup,a,b))
                end
            else
                push!(finalRoots,transformPoints(getFinalPoint(box),a,b))
            end
        end
    end
    # Find and return the roots (and, optionally, the bounding boxes)
    if returnBoundingBoxes
        return finalRoots, finalBoxes
    else
        return finalRoots
    end
end
