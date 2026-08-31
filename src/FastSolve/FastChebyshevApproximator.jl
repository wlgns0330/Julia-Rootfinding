import FFTW: r2r! #This is the DCT-I function that takes in a matrix and a transform "kind"
import FFTW: REDFT00 #This is the enum that represents DCT-I 
using Statistics

function fast_getApproxError(degs, epsilons, rhos, macheps=2^-52)
    """
    Computes an upper bound for the error of the Chebyshev approximation.

    Using the epsilon values and rates of geometric convergence calculated in getChebyshevDegrees,
    calculates the infinite sum of the coefficients past those used in the approximation.
    
    Parameters
    ----------
    degs : row array
        The degrees used in each dimension of the approximation.
    epsilons :  row array
        The values to which the approximation converged in each dimension.
    rhos : row array
        The calculated rate of convergence in each dimension.
    
    Returns
    -------
    approxError : Float64
        An upper bound on the approximation error
    """
    approxError = 0.0
    
    # Create a partition of coefficients where idxs[i]=1 represents coefficients being greater than
    # degs[i] in dimension i and idxs[i]=0 represents coefficients being less than [i] in dimension i.
    for idxs in Iterators.product(Iterators.repeated((0,1), length(degs))...)
        # Skip the set of all 0's, corresponding to the terms actually included in the approximation.
        if sum(idxs) == 0
            continue
        end
        
        s = 1.0
        thisEps = 0.0
        
        for (i, used) in enumerate(idxs)
            if Bool(used)
                # multiply by infinite sum of coeffs past the degree at which the approx stops in dim i
                #1/rhos[i] is the rate, so this is (1/rhos[i]) / (1 - 1/rhos[i]) = 1/(rhos[i]-1)
                s /= (rhos[i] - 1)
                # The points in this section are going to be < max(epsilons[i] that contribute to it)
                thisEps = max(thisEps, epsilons[i])
            else
                # multiply by the number of coefficients in the approximation along dim i
                s *= (degs[i] + 1)
            end
        end
        
        # Append to the error
        approxError += s * thisEps
    end

    return max(approxError,macheps)
end

function fast_transformPoints(x,a,b)
    """Transforms points from the interval [-1, 1] to the interval [a, b].

    Parameters
    ----------
    x : array
        The points to be tranformed. Each row is a point
    a : row array
        The lower bounds on the interval.
    b : row array
        The upper bounds on the interval.

    Returns
    -------
    transformed_pts : array
        The transformed points.
    """

    return ((b-a).*x .+(b+a))/2
end

function fast_getFinalDegree(coeff,tol,macheps = 2^-52)
    """Finalize the degree of Chebyshev approximation to use along one particular dimension.

    This function is called after the coefficients have started converging at degree n. A degree
    2n+1 approximation is passed in. Assuming that the coefficients have fully converged by degree 
    3n/2, the cutoff epsVal is calculated as twice the max coefficient of degree at least 3n/2.
    The final degree is then set as the largest coefficient with magnitude greater than epsVal.
    
    The rate of convergence is calculated assuming that the coefficients converge geometrically
    starting from the largest coefficient until machine epsilon is reached. This is a lower bound, as
    in practice, the coefficients usually slowly decrease at first but drop off fast at the end.

    Parameters
    ----------
    coeff : row array
        Absolute values of chebyshev coefficients.
    
    Returns
    -------
    degree : int
        The numerical degree of the approximation
    epsVal : float
        The epsilon value to which the coefficients have converged
    rho : float
        The geometric rate of convergence of the coefficients
    """

    # Set the final degree to the position of the last coefficient greater than convergence value
    converged_deg = Int64(div((3 * (length(coeff) - 1) / 4),1)) # Assume convergence at degree 3n/2.
    epsval = 2*max(macheps,maximum(coeff[converged_deg+1:end])) # Set epsVal to 2x the largest coefficient past degree 3n/2
    nonzero_coeffs_index = [i for i in 1:length(coeff) if coeff[i]>epsval]
    if isempty(nonzero_coeffs_index) 
        degree = 1
    else
        degree = max(1,nonzero_coeffs_index[end]-1)
    end

    # Set degree to 0 for constant functions (all coefficients but first are less than tol)
    if all(x -> x < tol, coeff[2:end])
        degree = 0
    end
    
    # Calculate the rate of convergence
    maxspot = argmax(coeff)
    if length(size(coeff)) > 1
        maxspot = maxspot[1]
    end
    if epsval == 0 #Avoid divide by 0. epsVal shouldn't be able to shrink by more than 1e-24 cause floating point.
         epsval = coeff[maxspot] * 1e-24
    end
    rho = (coeff[maxspot]/epsval)^(1/(degree - (maxspot) + 2))
    return degree, epsval, rho
end

function fast_startedConverging(coefflist,tol)
    """Determine whether the high-degree coefficients of a given Chebyshev approximation are near 0.

    Parameters
    ----------
    coeffList : row array
        Absolute values of chebyshev coefficients.
    tol : float
        Tolerance (distance from zero) used to determine whether coeffList has started converging.
    
    Returns
    -------
    startedConverging : bool
        True if the last 5 coefficients of coeffList are less than tol; False otherwise
    """
    return all(x -> x < tol, coefflist[end-4:end])
end

# `f` is deliberately not specialized on in the four functions below. They take the caller's
# function and pass it down, calling it only a handful of times each; specializing them recompiles
# a large amount of code for every new function a user writes, which is pure latency in an
# interactive session. `_fill_cheb_values!` -- the loop that calls `f` once per grid point -- IS
# still specialized, so the per-evaluation cost is unchanged. Measured: the compile cost of a new
# pair of functions drops from about 0.75s to 0.29s, with no loss of throughput.
function fast_checkConstantInDimension(@nospecialize(f),a,b,currdim,relTol,absTol=0)
    """Check to see if the output of f is not dependent on the input coordinate of a dimension.
    
    Uses predetermined random numbers to find a point x in the interval where f(x) != 0 and checks
    whether the value of f changes as the dimension currDim coordinate of x changes. Repeats twice.

    Parameters
    ----------
    f : function
        The function being evaluated.
    a : row array
        The lower bound on the interval.
    b : row array
        The upper bound on the interval.
    currDim : int
        The dimension being examined.
    
    Returns
    -------
    is_constant : bool
        Whether the dimension is constant in dimension currDim. Returns False if the test is
        indeterminate or f is seen to vary with different values of x[dim]. Returns True otherwise.
    """
    dim = length(a)
    currdim = currdim + 1
    # First test point x1
    x1 = fast_transformPoints([0.8984743990614998^(val) for val in 1:dim],a,b)
    eval1 = f(x1...)
    if isapprox(eval1,0,rtol=relTol,atol=absTol)
        return false
    end
    # Test how changing x_1[dim] changes the value of f for several values         
    for val in fast_transformPoints([-0.7996847717584993;0.18546110255464776;-0.13975937255055182;0.;1.;-1.],a[currdim],b[currdim])
        x1[currdim] = val
        eval2 = f(x1...)
        if !isapprox(eval1,eval2,rtol=relTol,atol=absTol) # Corresponding points gave different values for f(x)
            return false
        end
    end

    # Second test point x_2
    x2 = fast_transformPoints([(-0.2598647169391334*(val)/(dim))^2 for val in 1:dim],a,b)
    eval1 = f(x2...)
    if isapprox(eval1,0,rtol=relTol,atol=absTol) # Make sure f(x_2) != 0 (unlikely)
        return false
    end

    for val in fast_transformPoints([-0.17223860129797386;0.10828286380141305;-0.5333148248321931;0.46471703497219596],a[currdim],b[currdim])
        x2[currdim] = val
        eval2 = f(x2...)
        if !isapprox(eval1,eval2,rtol=relTol,atol=absTol)
            return false # Corresponding points gave different values for f(x)
        end
    end
    # Both test points had not zeros of f and had no variance along dimension currDim.
    return true
end

function fast_hasConverged(coeff, coeff2, tol)
    """Determine whether the high-degree coefficients of a Chebyshev approximation have converged
    to machine epsilon.

    Parameters
    ----------
    coeff : row array
        Absolute values of chebyshev coefficients of degree n approximation.
    coeff2 : row array
        Absolute values of chebyshev coefficients of degree 2n+1 approximation.
    tol : float
        Tolerance (distance from zero) used to determine whether the coefficients have converged.
    
    Returns
    -------
    hasConverged : Bool
        True if all the values of coeff and coeff2 are within tol of each other; False otherwise
    """
    coeff3 = copy(coeff2)
	coeff3[CartesianIndices(coeff)] .-= coeff 
    return maximum(abs.(coeff3)) < tol
end

function fast_intervalApproximateND(@nospecialize(f), degs, a, b, retSupNorm = false)
    """Generates an approximation of f on [a,b] using Chebyshev polynomials of degs degrees.

    Calculates the values of the function at the Chebyshev grid points and performs the FFT
    on these points to achieve the desired approximation.

    Parameters
    ----------
    f : function from R^n -> R
        The function to interpolate.
    a : row array
        The lower bound on the interval.
    b : row array
        The upper bound on the interval.
    degs : list of ints
        A list of the degree of interpolation in each dimension.
    retSupNorm : bool
        Whether to return the sup norm of the function.

    Returns
    -------
    coeffs : array
        The coefficients of the Chebyshev interpolating polynomial.
    supNorm : float (optional)
        The sup norm of the function, approximated as the maximum function evaluation.
    """
        dim = length(degs)
    # Dispatch on Val(dim) so the body specializes on the static dimension
    return _fast_intervalApproximateND_impl(f, degs, a, b, retSupNorm, Val(dim))
end

function _fast_intervalApproximateND_impl(@nospecialize(f), degs, a, b, retSupNorm::Bool,
                                          ::Val{N}) where {N}
    # If any dimension has degree 0, turn it to degree 1 (will be sliced out at the end)
    originalDegs = ntuple(d -> @inbounds(degs[d]), Val(N))
    anyZeroDeg = false
    @inbounds for i in 1:N
        if degs[i] == 0
            degs[i] = 1
            anyZeroDeg = true
        end
    end

    # Pre-compute Chebyshev grid points per dimension
    pts_per_dim = ntuple(Val(N)) do d
        deg = @inbounds degs[d]
        a_ = @inbounds a[d]
        b_ = @inbounds b[d]
        pd = pi / deg
        scale = (b_ - a_) / 2
        shift = (b_ + a_) / 2
        v = Vector{Float64}(undef, deg + 1)
        @inbounds for k in 0:deg
            v[k + 1] = scale * cos(k * pd) + shift
        end
        v
    end

    # Allocate output array with shape reverse(degs.+1) and fill directly with f(args...)
    out_shape = ntuple(d -> (@inbounds degs[N - d + 1]) + 1, Val(N))
    values = Array{Float64,N}(undef, out_shape)
    supNorm = _fill_cheb_values!(values, f, pts_per_dim, retSupNorm)

    # Perform Type-I DCT in place. FFTW's REDFT00 already absorbs the edge-halving on input.
    inv_prod = 1.0 / prod(degs)
    @inbounds @simd for i in eachindex(values)
        values[i] *= inv_prod
    end
    r2r!(values, REDFT00)
    coeffs = values

    # Halve edge slices along every dimension (must be done before reshape/transpose)
    _halve_edges!(coeffs, Val(N))

    if anyZeroDeg
        slices = ntuple(d -> 1:((@inbounds originalDegs[N - d + 1]) + 1), Val(N))
        coeffs = coeffs[slices...]
    end
    # Flatten 1D result back to a plain Vector so ndims(coeffs) == 1, matching the pre-refactor
    # return type. Returning a (1,N) Adjoint here instead would make downstream code (e.g.
    # fast_transformChebInPlace1D's ndims(coeffs)==1 fast path) skip the univariate fast path.
    if N == 1
        return retSupNorm ? (vec(coeffs), supNorm) : vec(coeffs)
    end
    return retSupNorm ? (coeffs, supNorm) : coeffs
end

@generated function _gather_args(pts::NTuple{N,Vector{T}}, J::CartesianIndex{N}) where {N,T}
    return Expr(:tuple, [:(@inbounds pts[$d][J.I[$(N - d + 1)]]) for d in 1:N]...)
end

@inline function _fill_cheb_values!(values::AbstractArray{T,N}, f::F,
                                    pts_per_dim::NTuple{N,Vector{T}}, retSupNorm::Bool) where {T,N,F}
    sup = zero(T)
    @inbounds for J in CartesianIndices(values)
        args = _gather_args(pts_per_dim, J)
        v = f(args...)
        values[J] = v
        if retSupNorm
            av = abs(v)
            if av > sup
                sup = av
            end
        end
    end
    return sup
end

@inline function _halve_edges!(coeffs::AbstractArray{T,N}, ::Val{N}) where {T,N}
    @inbounds for ax in 1:N
        n = size(coeffs, ax)
        first_slice = ntuple(k -> k == ax ? (1:1) : axes(coeffs, k), Val(N))
        last_slice  = ntuple(k -> k == ax ? (n:n) : axes(coeffs, k), Val(N))
        @views coeffs[first_slice...] ./= 2
        @views coeffs[last_slice...]  ./= 2
    end
    return coeffs
end

function fast_createMeshgrid(arrays...)
    """ Takes arguments x1,x2,...,xn, each xi being a 'row' vector. Does meshgrid. 
    Output is in the format [X,Y,Z,...], where each element in the list is a matrix.
    Example: createMeshgrid([1;2],[3;4]) -> [[1;1;;2;2],[3;4;;3;4]].
        Note: This would output as [[1 2;1 2],[3 3;4 4]], which looks wrong, but for our 
        purposes it will be easier to think of the matricies in the first format instead of the printing one.
        This way, accessing elements and slices is exactly like in Python but reversed.
    """
    dims = []
    for array in arrays
        push!(dims,length(array))
    end
    finals = []
    for iter in range(1,length(arrays))
        # Get product of array sizes before and after the current array
        reps = 1
        full_reps=1
        try 
            reps = prod(dims[iter+1:end])
        catch end
        try 
            full_reps = prod(dims[1:iter-1])
        catch end
        # Repeat the current array into a matrix with height and width determined by the sizes before and after
        newArray = arrays[iter]
        endArray = repeat(arrays[iter],full_reps,reps)
        # Reshape it to the meshgrid array and push it onto our final list
        push!(finals,reshape(endArray',Tuple(reverse(dims)))) 
    end
    return finals
end

function fast_getChebyshevDegrees(@nospecialize(f), a, b, relApproxTol, absApproxTol = 0)
    """Compute the minimum degrees in each dimension that give a reliable Chebyshev approximation for f.

    For each dimension, starts with degree 8, generates an approximation, and checks to see if the
    sequence of coefficients is converging. Repeats, doubling the degree guess until the coefficients
    are seen to converge to 0. Then calls getFinalDegree to get the exact degree of convergence.
    
    Parameters
    ----------
    f : function
        The function being approximated.
    a : array-like
        The lower bound on the interval.
    b : array-like
        The upper bound on the interval.
    relApproxTol : float
        The relative tolerance (distance from zero) used to determine convergence
    absApproxTol : float
        The absolute tolerance (distance from zero) used to determine convergence
    
    Returns
    -------
    chebDegrees : array-like
        The numerical degree in each dimension.
    epsilons : array-like
        The value the coefficients converged to in each dimension.
    rhos : array-like
        The rate of convergence in each dimension.
    """
    a = reshape(a,:,1)
    b = reshape(b,:,1)
    dim = length(a)
    chebDegrees = ones(Int64,dim,1)*Inf # the approximation degree in each dimension
    epsilons = [] # the value the approximation has converged to in each dimension
    rhos = [] # the calculated rate of convergence in each dimension
    # Check to see if f varies each input; set degree to 0 if not
    for currDim in range(1,dim)
        if fast_checkConstantInDimension(f,a,b,currDim-1,relApproxTol,absApproxTol)
            chebDegrees[currDim] = 0
        end
    end
    # Find the degree in each dimension seperately
    for currDim in range(1,dim)
        if chebDegrees[currDim] == 0 # skip the guessing algorithm if f is constant in dim currDim
            push!(epsilons,0)
            push!(rhos,Inf)
            continue
        end
        # Isolate the current dimension by fixing all other dimensions at constant degree approximation
        degs = ones(Int,dim,1)*(dim<=5 ? 5 : 2)
        for i in range(1,dim) # save computation by using already computed degrees if lower
            if chebDegrees[i] < degs[i]
                degs[i] = chebDegrees[i]
            end
        end
        currGuess = 8 # Take initial guess degree 8 in the current dimension
        tupleForChunk = tuple(deleteat!([i for i in range(1,dim)],dim+1-currDim)...)
        while true # Runs until the coefficients are shown to converge to 0 in this dimension
            if currGuess > 1e5
                #warnings.warn(f"Approximation bound exceeded!\n\nApproximation degree in dimension {currDim} "
                #              + "has exceeded 1e5, so the process may not finish.\n\nConsider interrupting "
                #              "and restarting the process after ensuring that the function(s) inputted are " +
                #              "continuous and smooth on the approximation interval.\n\n")
            end
            #g = lambda *x: f(*x)/totSupNorm
            degs[currDim] = currGuess
            coeff, supNorm = fast_intervalApproximateND(f, degs, a, b, true) # get approximation
            #totSupNorm *= supNorm
            # Get "average" coefficients along the current dimension
            coeffChunk = mean(abs.(coeff), dims=tupleForChunk)
            tol = absApproxTol + supNorm * relApproxTol # Set tolerance for convergence from the supNorm
            currGuess *= 2 # Ensure the degree guess is doubled in case of another iteration

            # Check if the coefficients have started converging; iterate if they have not.
            if !fast_startedConverging(coeffChunk, tol)
                continue
            end
            # Since the coefficients have started to converge, check if they have fully converged.
            # Degree n and 2n+1 are unlikely to have higher degree terms alias into the same spot.
            degs[currDim] = currGuess + 1 # 2n+1
            coeff2, supNorm2 = fast_intervalApproximateND(f, degs, a, b, true)
            tol = absApproxTol + max(supNorm, supNorm2) * relApproxTol
            if !hasConverged(coeff, coeff2, tol)
                continue # Keed doubling if the coefficients have not fully converged.
            end
            # The coefficients have been shown to converge to 0. Get the exact degree where this occurs.
            coeffChunk = reshape(mean(abs.(coeff2), dims=tupleForChunk),(:,1))
            deg, eps, rho = fast_getFinalDegree(coeffChunk,tol)
            chebDegrees[currDim] = deg
            push!(epsilons,eps)
            push!(rhos,rho)
            break # Finished with the current dimension
        end
    end
    return Int.(chebDegrees), epsilons, rhos
end

function fast_chebApproximate(@nospecialize(f), a, b, relApproxTol=1e-10)
    # TODO:implement a way for the user to input Chebyshev coefficients they may already have, (MultiCheb/MultiPower stuff in python implementation)
    """

    Parameters
    ----------
    f : function
        The function to be approximated. NOTE: Valid input is restricted to callable Python functions
        (including user-created functions) and yroots Polynomial (MultiCheb and MultiPower) objects.
        String representations of functions are not valid input.
    a: array-like or single value
        An array containing the lower bound of the approximation interval in each dimension, listed in
        dimension order
    b: array-like or single value
        An array containing the upper bound of the approximation interval in each dimension, listed in
        dimension order.
    relApproxTol : float
        The relative tolerance used to determine at what degree the Chebyshev coefficients have
        converged to zero. If all coefficients after degree n are within relApproxTol * supNorm
        (the maximum function evaluation on the interval) of zero, the coefficients will be
        considered to have converged at degree n. Defaults to 1e-10.
    
    Returns
    -------
    coefficient_matrix : numpy array
        The coefficient matrix of the Chebyshev approximation.
    error : float
        The error associated with the approximation.
    """
    # Convert single values to arrays. This is in the case that a, b ∈ ℝ
	a = (a isa Real) ? [a] : a
	b = (b isa Real) ? [b] : b

	if length(a) ≠ length(b)
		throw(ArgumentError("Invalid input: $(length(a)) lower bounds were given but $(length(b)) upper bounds were given"))
	end
    
	#TODO: Figure out if this code runs through the whole array first or not .< may be slower than checking each element individually
	if any(b .< a)
		throw(ArgumentError("Invalid input: at least one lower bound is greater than the corresponding upper bound."))
	end

    try
        f(a...)
    catch
        throw(ArgumentError("Invalid input: length of the upper/lower bound lists does not match the dimension (no. inputs) of the function"))
    end
    # Generate and return the approximation
    degs, epsilons, rhos = fast_getChebyshevDegrees(f, a, b, relApproxTol)
    return fast_intervalApproximateND(f, degs, a, b), fast_getApproxError(degs, epsilons, rhos)
end