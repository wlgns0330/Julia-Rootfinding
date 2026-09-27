
"""Finds and returns the roots of a system of functions on the search interval [a,b].

Generates an approximation for each function using Chebyshev polynomials on the interval given,
then uses properties of the approximations to shrink the search interval. When the information
contained in the approximation is insufficient to shrink the interval further, the interval is
subdivided into subregions, and the searching function is recursively called until it zeros in
on each root. A specific point (and, optionally, a bounding box) is returned for each root found.

NOTE: YRoots uses just in time compiling, which means that part of the code will not be compiled until
a system of functions to solve is given (rather than compiling all the code upon importing the module).
As a result, the very first time the solver is given any system of equations of a particular dimension,
the module will take several seconds longer to solve due to compiling time. Once the first system of a
particular dimension has run, however, other systems of that dimension (or even the same system run
again) will be solved at the normal (faster) speed thereafter.

NOTE: The solve function is only guaranteed to work well on systems of equations where each function
is continuous and smooth and each root in the interval is a simple root. If a function is not
continuous and smooth on an interval or an infinite number of roots exist in the interval, the
solver may get stuck in recursion or the kernel may crash.

Examples
--------

>>> f = lambda x,y,z: 2*x**2 / (x**4-4) - 2*x**2 + .5
>>> g = lambda x,y,z: 2*x**2*y / (y**2+4) - 2*y + 2*x*z
>>> h = lambda x,y,z: 2*z / (z**2-4) - 2*z
>>> roots = yroots.solve([f, g, h], np.array([-0.5,0,-2**-2.44]), np.array([0.5,np.exp(1.1376),.8]))
>>> print(roots)
[[-4.46764373e-01  4.44089210e-16 -5.55111512e-17]
 [ 4.46764373e-01  4.44089210e-16 -5.55111512e-17]]



>>> M1 = yroots.MultiPower(np.array([[0,3,0,2],[1.5,0,7,0],[0,0,4,-2],[0,0,0,1]]))
>>> M2 = yroots.MultiCheb(np.array([[0.02,0.31],[-0.43,0.19],[0.06,0]]))
>>> roots = yroots.solve([M1,M2],-5,5)
>>> print(roots)
[[-0.98956615 -4.12372817]
 [-0.06810064  0.03420242]]

Parameters
----------
funcs: list
    List of functions for searching. NOTE: Valid input is restricted to callable Python functions
    (including user-created functions) and yroots Polynomial (MultiCheb and MultiPower) objects.
    String representations of functions are not valid input.
a: list or numpy array
    An array containing the lower bound of the search interval in each dimension, listed in
    dimension order. If the lower bound is to be the same in each dimension, a single float input
    is also accepted. Defaults to -1 in each dimension if no input is given.
b: list or numpy array
    An array containing the upper bound of the search interval in each dimension, listed in
    dimension order. If the upper bound is to be the same in each dimension, a single float input
    is also accepted. Defaults to 1 in each dimension if no input is given.
verbose : bool
    Defaults to False. Tracks progress of the approximation and rootfinding by outputting progress to
    the terminal. Useful in tracking progress of systems of equations that take a long time to solve.
returnBoundingBoxes : bool
    Defaults to False. Whether or not to return a precise bounding box for each root.
exact: bool
    Defaults to False. Whether transformations performed on the approximation should be performed
    with higher precision to minimize error.
_refine : bool
    Internal. Whether to solve again around a root whose box stayed wide (see _refineWideBoxes).
    False on the solves that refinement itself makes, so it happens once.
_maxDegree : int or nothing
    Internal. The highest degree a callable's approximation may take (see REFINE_MAX_DEGREE).
minBoundingIntervalSize : double
    Defaults to 1e-5. If a root is found with a bounding interval of size > minBoundingIntervalSize in
    each dimension, the functions are solved again on the smaller interval. Setting too small could cause
    issues if the functions can't be evaluated accurately on points close together, and will increase solve
    times. Should give more accurate roots when smaller. This number is absolute when the boudning interval in
    question is in [-1,1], and relative otherwise. So if an interval has an endpoint of magnitude > 1, then
    minBoundingIntervalSize is multipled by that value for that dimension.

Returns
-------
yroots : numpy array
    A list of the roots of the system of functions on the interval.
boundingBoxes : numpy array (optional)
    The exact intervals (boxes) in which each root is bound to lie.
"""
# A root whose final box is wider than this, relative to the size of the search interval's bounds in that
# dimension (as minBoundingIntervalSize is), converged the slow way an ill-conditioned root does; a simple
# root's box ends up around 1e-13. Such a box is solved again on a padded neighborhood of itself.
const REFINE_BOX_SIZE = 1e-10
# The neighborhood re-solved around a wide box reaches this many of the box's widths past it on each side.
const REFINE_PADDING = 2
# An approximation on that neighborhood may not need a degree above this. A smooth function needs only a
# handful of coefficients on so small an interval. One whose rounding error is large next to its values
# there (1 - cos(y) near 0, where the cancellation happens inside the function) never converges and
# keeps doubling its degree, so this is where refinement gives up and the box keeps what it had.
const REFINE_MAX_DEGREE = 100

"""
The point where a search interval `[a, b]` that could not be shrunk is split, `frac` of the way from
`a` to `b` in each dimension; `frac` is just off one half, so a root on the exact midpoint does not land
on the split. Computed from the width, so it stays inside the interval wherever the interval is: scaling
`a + b` instead gave a point outside it for an interval away from the origin: -101.96 for `[-100, -99]`.
"""
subdivisionPoint(a, b, frac) = a .+ (b .- a) .* frac

"""
Whether `[a, b]` is the box [-1, 1]^n that a polynomial's coefficients are written on. Only there are
MultiPower and MultiCheb coefficients used as they are; on any other box, including the sub-boxes a
solve recurses on, the polynomial is approximated there like any callable.
"""
isUnitBox(a, b) = all(a .== -1) && all(b .== 1)

"""
Whether every function is a polynomial given as coefficients, a system `_refineWideBoxes` cannot improve.

Evaluating a polynomial from its coefficients is off by about macheps times the sum of its terms' sizes,
however close to a root, so its roots are only as good as that error allows (about sqrt of it, for a
near-double root) and approximating it again on a neighborhood of a root just raises the degree until
REFINE_MAX_DEGREE: on Chebfun2 case 1.2 given as a MultiPower or MultiCheb, every wide box does. A
callable such as a product of factors can be accurate relative to its value, and is refined.
"""
allPolynomial(funcs) = all(f isa MultiPower || f isa MultiCheb for f in funcs)

"""
The roots and boxes one final box of the subdivision solve contributes, in the original coordinates,
as an entry `(roots, boxes, isWide)` for `_refineWideBoxes`.

The box is repeated once per root, so the roots and boxes a solve returns stay paired even when the
final step keeps several possible duplicates in one box. `isWide` marks a box wider than `wideSize`
relative to `scale`.
"""
function finalBoxEntry(box, a, b, transformPoints, getFinalPoint, scale, wideSize)
    transformedBox = transformPoints(box.finalInterval', a, b)'
    boxRoots = Any[]
    if length(box.possibleDuplicateRoots) > 0
        for dup in box.possibleDuplicateRoots
            push!(boxRoots, transformPoints(dup, a, b))
        end
    else
        push!(boxRoots, transformPoints(getFinalPoint(box), a, b))
    end
    isWide = maximum((transformedBox[2,:] .- transformedBox[1,:]) ./ scale) > wideSize
    return (boxRoots, Any[transformedBox for _ in boxRoots], isWide)
end

"""
Solve again around each root whose final box is wide, and replace that box's roots with what is found.

Two roots closer than about sqrt(macheps) times the width of the interval an approximation was built
on are one dip below that approximation's error, so they come back as one point, or a pair straddling
the dip. Neither the error nor the dip is set by the roots: the dip is (d/2)^2 times the function's
curvature, and the error shrinks with the function's size on the interval. Approximating again on a
small neighborhood of the box therefore separates roots that are far closer together, whenever the
function can be evaluated that accurately near them (see allPolynomial for when it cannot).

`solver` is the solve to run on each neighborhood, `fast_solve` or the general-precision `solve`,
called with `kwargs`. `entries` holds one `(roots, boxes, isWide)` tuple per final box, in the original
coordinates, with each box a 2 x dim array whose first row is the lower bound and one box per root.
The roots and boxes of each wide entry are replaced. An entry keeps what it had when its neighborhood
cannot be solved or turns up no root; a neighborhood this small is at the edge of what the solver
handles: y^2 = 0 on one, for one, reports no root at all, and a function that cannot be evaluated
accurately there needs a degree above REFINE_MAX_DEGREE.
"""
function _refineWideBoxes(solver, funcs, a, b, entries; kwargs...)
    scale = max.(abs.(a), abs.(b), 1)
    # Every box in units of scale, and the entry it came from, to decide which entry a root found on a
    # neighborhood belongs to.
    allLower = [box[1,:] ./ scale for entry in entries for box in entry[2]]
    allUpper = [box[2,:] ./ scale for entry in entries for box in entry[2]]
    boxOwner = [k for (k, entry) in enumerate(entries) for _ in entry[2]]
    # The entry nearest a point, measured in units of scale.
    nearestEntry(p) = boxOwner[argmin([maximum(max.(lo .- p ./ scale, p ./ scale .- hi))
                                       for (lo, hi) in zip(allLower, allUpper)])]
    for k in eachindex(entries)
        roots, boxes, isWide = entries[k]
        isWide || continue
        box = boxes[1]
        # Pad in units of the box's widest side in every dimension, so a dimension that already converged
        # to a subnormal width is still a neighborhood worth approximating on.
        halfWidth = REFINE_PADDING * maximum((box[2,:] .- box[1,:]) ./ scale) .* scale
        newA = max.(box[1,:] .- halfWidth, a)
        newB = min.(box[2,:] .+ halfWidth, b)
        newRoots, newBoxes = try
            solver(funcs, newA, newB; returnBoundingBoxes=true, _refine=false,
                   _maxDegree=REFINE_MAX_DEGREE, kwargs...)
        catch e
            (e isa DegreeCapExceeded || e isa StackOverflowError) || rethrow()
            continue
        end
        # The neighborhood can reach into another box; a root that belongs to that box is left to it.
        # Roots and boxes come back paired, one box per root, so they are kept or dropped together.
        own = [nearestEntry(r) == k for r in newRoots]
        any(own) || continue
        entries[k] = (newRoots[own], newBoxes[own], false)
    end
    return entries
end

function fast_solve(funcs,a,b; verbose, returnBoundingBoxes, exact, minBoundingIntervalSize,
                    _refine=true, _maxDegree=nothing)
    dim = length(funcs)
    # polys = Vector{Array{Float64}}()
    polys = Vector{Array{Float64,dim}}(undef, dim)
    errs = fill(0.0,dim)
    # Get an approximation for each function.
    if verbose
        print("Approximation shapes:")
        print(" ")
    end

    unitBox = isUnitBox(a, b)
    for i in 1:dim
        if funcs[i] isa MultiPower && unitBox
            polys[i] = multipower_to_cheb(funcs[i].coeff)
            errs[i] = polynomialRoundingError(polys[i], 2. ^-52)
        elseif funcs[i] isa MultiCheb && unitBox
            polys[i] = funcs[i].coeff
            errs[i] = polynomialRoundingError(polys[i], 2. ^-52)
        else
            polys[i], errs[i] = fast_chebApproximate(funcs[i],a,b; maxDegree=_maxDegree)
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
    yroots, boundingBoxes = fast_solveChebyshevSubdivision(polys,errs;verbose=verbose,returnBoundingBoxes=true,exact=exact,
                constant_check=true, low_dim_quadratic_check=true, all_dim_quadratic_check=false)

    #If the bounding box is the entire interval, subdivide it!
    usingSubdivision = all(b-a .> minBoundingIntervalSize)
    if length(boundingBoxes) == 1 && all(fast_finalDimSize(boundingBoxes[1]) .== 2) && usingSubdivision
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
            roots, boxes = fast_solve(funcs, newA, newB; verbose=verbose, returnBoundingBoxes=true, exact=exact, minBoundingIntervalSize = minBoundingIntervalSize,
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
    for box in boundingBoxes
        #Get the relative max size in each dimension. If a or b > 1 in magnitude, minBoundingIntervalSize is a relative number.
        #If they are < 1 in magnitude, it is an absolute number.
        newBox = fast_transformPoints(box.finalInterval',a,b)
        newA, newB = newBox[:,1],newBox[:,2]

        relMaxSize = minBoundingIntervalSize .* maximum(hcat(abs.(a),abs.(b),fill(1,length(a))),dims=2)
        if all(newB - newA .> relMaxSize)
            #Re-solve this box
            if verbose
                print("Re-solving on: ")
                print(newA)
                print(" ")
                println(newB)
            end
            roots, boxes = fast_solve(funcs, newA, newB; verbose=verbose, returnBoundingBoxes=true, exact=exact, minBoundingIntervalSize = minBoundingIntervalSize,
                                      _refine=_refine, _maxDegree=_maxDegree)
            if length(roots) > 0
                push!(entries, (collect(roots), collect(boxes), false))
            end
        else
            push!(entries, finalBoxEntry(box, a, b, fast_transformPoints, fast_getFinalPoint, scale, REFINE_BOX_SIZE))
        end
    end

    if _refine && any(entry[3] for entry in entries) && !allPolynomial(funcs)
        entries = _refineWideBoxes(fast_solve, funcs, a, b, entries; verbose=verbose, exact=exact,
                                   minBoundingIntervalSize=minBoundingIntervalSize)
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
