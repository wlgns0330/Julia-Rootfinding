mutable struct FastTrackedInterval 
    """Tracks the properties of and changes to each interval as it passes through the solver.

    Parameters
    ----------
    topInterval: array
        The original interval before any changes
    interval: array
        The current interval (lower bound and upper bound for each dimension in order)
    transforms: array
        List of the alpha and beta values for all the transformations the interval has undergone
    ndim: Int
        The number of dimensions of which the interval consists
    empty: bool
        Whether the interval is known to contain no roots
    finalStep: bool
        Whether the interval is in the final step (zooming in on the bounding box to a point at the end)
    canThrowOutFinalStep: bool
        Defaults to false. Whether or not the interval should be thrown out if empty in the final step
        of solving. Changed to true if subdivision occurs in the final step.
    possibleDuplicateRoots: array
        Any multiple roots found through subdivision in the final step that would have been
        returned as just one root before the final step
    possibleExtraRoot: bool
        Defaults to false. Whether or not the interval would have been thrown out during the final step.
    nextTransformPoints: array
        Where the midpoint of the next subdivision should be for each dimension
    """

    # This struct is implemented by passing in one argument "interval"
    # eg: TrackedInterval([-1;-3.4;0])
    topInterval::Matrix{Float64}
    interval::Matrix{Float64}
    transforms::Vector{Matrix{Float64}}
    ndim::Int
    empty::Bool
    finalStep::Bool
    canThrowOutFinalStep::Bool
    possibleDuplicateRoots::Vector{Vector{Float64}}
    possibleExtraRoot::Bool
    nextTransformPoints::Vector{Float64}
    preFinalInterval::Matrix{Float64}
    preFinalTransforms::Vector{Matrix{Float64}}
    reducedDims::Vector{Int}
    solvedVals::Vector{Float64}
    finalInterval::Matrix{Float64}
    finalAlpha::Vector{Float64}
    finalBeta::Vector{Float64}
    reRun::Bool
    root::Vector{Float64}
    function FastTrackedInterval(interval::AbstractMatrix)
        m = Matrix{Float64}(interval)
        ndim = Int(length(m)/2)
        empty_mat = Matrix{Float64}(undef, 0, 0)
        new(m, copy(m), Matrix{Float64}[], ndim, false, false, false,
            Vector{Float64}[], false, fill(0.0394555475981047, ndim),
            empty_mat, Matrix{Float64}[], Int[], Float64[], empty_mat,
            Float64[], Float64[], false, Float64[])
    end
end

"""==============================FUNCTIONS FOR TRACKED INTERVAL=============================="""

function fast_canThrowOut(trackedInterval::FastTrackedInterval)
    """Ensures that an interval that has not subdivided cannot be thrown out on the final step."""
    return !trackedInterval.finalStep || trackedInterval.canThrowOutFinalStep
end

function fast_addTransform(trackedInterval::FastTrackedInterval, subInterval)
    """Adds the next alpha and beta values to the list transforms and updates the current interval.

    Parameters:
    -----------
    subInterval : array
        The subinterval to which the current interval is being reduced
    """
    #Ensure the interval has non zero size; mark it empty if it doesn't
    # NOTE: `subInterval[1,:] > subInterval[2,:]` compares the two rows lexicographically and
    # yields a single Bool, so `any` of it is just that Bool. Preserved as-is; only the row
    # copies it made are gone.
    degenerate = fast_rowGreater(subInterval)
    if degenerate && fast_canThrowOut(trackedInterval)
        trackedInterval.empty = true
        return
    elseif degenerate
        #If we can't throw the interval out, it should be bounded by [-1,1].
        subInterval[1,:] = min.(subInterval[1,:], ones(length(subInterval[1,:])))
        subInterval[1,:] = max.(subInterval[1,:], -ones(length(subInterval[1,:])))
        subInterval[2,:] = min.(subInterval[2,:], ones(length(subInterval[1,:])))
        subInterval[2,:] = max.(subInterval[2,:], subInterval[1,:])
    end
    # Get the alpha and beta associated with the transformation in each dimension. Built straight
    # into the stored ndim x 2 matrix instead of through six intermediate row vectors and an hcat.
    n = trackedInterval.ndim
    iv = trackedInterval.interval
    transform = Matrix{Float64}(undef, n, 2)
    @inbounds for d in 1:n
        a1 = subInterval[1, d]
        b1 = subInterval[2, d]
        transform[d, 1] = (b1 - a1) / 2.
        transform[d, 2] = (b1 + a1) / 2.
    end
    push!(trackedInterval.transforms, transform)
    #Update the lower and upper bounds of the current interval
    @inbounds for d in 1:n
        a2 = iv[1, d]
        b2 = iv[2, d]
        alpha2 = (b2 - a2) / 2.
        beta2 = (b2 + a2) / 2.
        for i in 1:2
            x = subInterval[i, d]
            #Be exact if x = +-1
            if x == -1.0
                iv[i, d] = a2
            elseif x == 1.0
                iv[i, d] = b2
            else
                iv[i, d] = alpha2 * x + beta2
            end
        end
    end
end

"""Lexicographic `subInterval[1,:] > subInterval[2,:]`, without materializing the two rows.

Comparing with `isless` rather than `==`/`>` matters: vector `>` is `isless` under the hood, and
`isless` orders -0.0 below 0.0 where `==` calls them equal."""
function fast_rowGreater(subInterval)
    @inbounds for d in 1:size(subInterval, 2)
        lo = subInterval[1, d]
        hi = subInterval[2, d]
        isless(hi, lo) && return true
        isless(lo, hi) && return false
    end
    return false
end

function fast_getLastTransform(trackedInterval::FastTrackedInterval)
    return trackedInterval.transforms[end]
end

"""Replays a list of saved transforms onto the top interval, tracking the rounding error.

Returns the transformed interval and its error, both as 2 x ndim matrices (bound, dimension).
The arithmetic is the same error-free product and sum the array version did, done one element at
a time so nothing is allocated per transform. Transform lists get long on deep subdivisions, and
the array version allocated roughly a dozen 2 x ndim temporaries for each entry."""
function fast_replayTransforms(topInterval::Matrix{Float64}, transforms::Vector{Matrix{Float64}})
    n = Int(length(topInterval) / 2)
    interval = Matrix{Float64}(undef, 2, n)
    err = zeros(2, n)
    @inbounds for d in 1:n, c in 1:2
        interval[c, d] = topInterval[c, d]
    end
    @inbounds for t in length(transforms):-1:1
        transform = transforms[t]
        for d in 1:n
            alpha = transform[d, 1]
            beta = transform[d, 2]
            for c in 1:2
                x, temp = fast_twoProdScalar(interval[c, d], alpha)
                interval[c, d] = x
                e = alpha * err[c, d] + temp
                x, temp = fast_twoSumScalar(interval[c, d], beta)
                interval[c, d] = x
                err[c, d] = e + temp
            end
        end
    end
    return interval, err
end

function fast_getFinalInterval(trackedInterval::FastTrackedInterval)
    """Finds the point that should be reported as the root (midpoint of the final step interval).

    Returns
    -------
    root: numpy array
        The final point to be reported as the root of the interval
    """
    transformsToUse = trackedInterval.finalStep ? trackedInterval.preFinalTransforms : trackedInterval.transforms
    finalInterval, finalIntervalError = fast_replayTransforms(trackedInterval.topInterval, transformsToUse)
    n = size(finalInterval, 2)
    trackedInterval.finalInterval = finalInterval + finalIntervalError # Add the error and save the result.
    finalAlpha = Vector{Float64}(undef, n)
    finalBeta = Vector{Float64}(undef, n)
    @inbounds for d in 1:n
        lo = finalInterval[1, d] / 2.
        hi = finalInterval[2, d] / 2.
        elo = finalIntervalError[1, d]
        ehi = finalIntervalError[2, d]
        alpha, alphaError = fast_twoSumScalar(-lo, hi)
        finalAlpha[d] = alpha + (alphaError + (ehi - elo) / 2.)
        beta, betaError = fast_twoSumScalar(lo, hi)
        finalBeta[d] = beta + (betaError + (ehi + elo) / 2.)
    end
    trackedInterval.finalAlpha = finalAlpha
    trackedInterval.finalBeta = finalBeta
    return trackedInterval.finalInterval
end

function fast_getFinalPoint(trackedInterval::FastTrackedInterval)
    """Finds the point that should be reported as the root (midpoint of the final step interval).

    Returns
    -------
    root: numpy array
        The final point to be reported as the root of the interval
    """
    if !trackedInterval.finalStep  # If no final step, use the midpoint of the calculated final interval.
        fi = trackedInterval.finalInterval
        n = size(fi, 2)
        root = Vector{Float64}(undef, n)
        @inbounds for d in 1:n
            root[d] = (fi[1, d] + fi[2, d]) / 2.
        end
        trackedInterval.root = root
    else  # If using the final step, recalculate the final interval using post-final transforms.
        finalInterval, finalIntervalError = fast_replayTransforms(trackedInterval.topInterval,
                                                                 trackedInterval.transforms)
        n = size(finalInterval, 2)
        root = Vector{Float64}(undef, n)
        @inbounds for d in 1:n
            lo = finalInterval[1, d] + finalIntervalError[1, d]
            hi = finalInterval[2, d] + finalIntervalError[2, d]
            root[d] = (lo + hi) / 2.  # Return the midpoint
        end
        trackedInterval.root = root
    end
    return trackedInterval.root
end

# not thoroughly tested
function fast_sizeOfInterval(trackedInterval)
    """Gets the volume of the current interval."""
    iv = trackedInterval.interval
    v = 1.0
    @inbounds for d in 1:size(iv, 2)
        v *= iv[2, d] - iv[1, d]
    end
    return v
end

function fast_dimSize(trackedInterval)
    """Gets the lengths along each dimension of the current interval."""
    iv = trackedInterval.interval
    n = size(iv, 2)
    out = Vector{Float64}(undef, n)
    @inbounds for d in 1:n
        out[d] = iv[2, d] - iv[1, d]
    end
    return out
end

function fast_finalDimSize(trackedInterval)
    """Gets the lengths along each dimension of the current interval."""
    iv = trackedInterval.finalInterval
    n = size(iv, 2)
    out = Vector{Float64}(undef, n)
    @inbounds for d in 1:n
        out[d] = iv[2, d] - iv[1, d]
    end
    return out
end

function fast_copyInterval(trackedInterval::FastTrackedInterval)
    """Returns a deep copy of the current interval with all changes and properties preserved."""
    newone = FastTrackedInterval(trackedInterval.topInterval)
    newone.interval = copy(trackedInterval.interval)
    newone.transforms = copy(trackedInterval.transforms)
    newone.empty = trackedInterval.empty
    newone.nextTransformPoints = copy(trackedInterval.nextTransformPoints)
    if trackedInterval.finalStep
        newone.finalStep = true
        newone.canThrowOutFinalStep = trackedInterval.canThrowOutFinalStep
        newone.possibleDuplicateRoots = copy(trackedInterval.possibleDuplicateRoots)
        newone.possibleExtraRoot = trackedInterval.possibleExtraRoot
        newone.preFinalInterval = copy(trackedInterval.preFinalInterval)
        newone.preFinalTransforms = copy(trackedInterval.preFinalTransforms)
    end
    return newone
end

# Not tested or used
function fast_contains(trackedInterval::FastTrackedInterval, point)
    """Determines if point is contained in the current interval."""
    return all(point >= trackedInterval.interval[1,:]) && all(point <= trackedInterval.interval[2,:])
end

function fast_overlapsWith(trackedInterval::FastTrackedInterval, otherInterval::FastTrackedInterval)
    """Determines if the otherInterval overlaps with the current interval.

    Returns True if the lower bound of one interval is less than the upper bound of the other
        in EVERY dimension; returns False otherwise."""
    currentInterval = fast_getIntervalForCombining(trackedInterval)
    otherInterval = fast_getIntervalForCombining(otherInterval)
    size_arr = size(currentInterval)
    if length(size_arr) == 1
        dim = 1
    else
        dim = size_arr[2]
    end

    arr1_1 = currentInterval[1,:]
    arr1_2 = currentInterval[2,:]
    arr2_1 = otherInterval[1,:]
    arr2_2 = otherInterval[2,:]

    for i in 1:dim
        if ((arr1_1[i] > arr2_2[i]) || (arr2_1[i] > arr1_2[i]))
            return false
        end
    end
    return true
end

function fast_isPoint(trackedInterval::FastTrackedInterval, macheps = 2. ^-52)
    """Determines if the current interval has essentially length 0 in each dimension."""
    iv = trackedInterval.interval
    @inbounds for d in 1:size(iv, 2)
        abs(iv[1, d] - iv[2, d]) < macheps || return false
    end
    return true
end

function fast_startFinalStep(trackedInterval::FastTrackedInterval)
    """Prepares for the final step by saving the current interval and its transform list."""
    trackedInterval.finalStep = true
    trackedInterval.preFinalInterval = copy(trackedInterval.interval)
    trackedInterval.preFinalTransforms = copy(trackedInterval.transforms)
end

function fast_getIntervalForCombining(trackedInterval::FastTrackedInterval)
    """Returns the interval to be used in combining intervals to report at the end."""
    return trackedInterval.finalStep ? trackedInterval.preFinalInterval : trackedInterval.interval
end

function fast_toStr(trackedInterval::FastTrackedInterval)
    return string(trackedInterval.interval)
end
