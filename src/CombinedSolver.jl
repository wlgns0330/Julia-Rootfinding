include("ChebyshevApproximator.jl")
include("ChebyshevSubdivisionSolver.jl")
include("StructsWithTheirFunctions/Polynomial.jl")
include("FastSolve/FastCombinedSolver.jl")


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
