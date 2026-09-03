mutable struct TrackedInterval 
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
    topInterval # = interval (by default) 
    interval # = interval (by default) 
    transforms # = [] (by default)
    ndim # = length(interval) (by default)
    empty # = false (by default)
    finalStep # = false (by default)
    canThrowOutFinalStep # = false (by default)
    possibleDuplicateRoots # = [] (by default)
    possibleExtraRoot # = false (by default)
    nextTransformPoints #Random Point near 0
    preFinalInterval # = [] (by default)
    preFinalTransforms # = [] (by default)
    reducedDims # = [] (by default)
    solvedVals # = [] (by default)
    finalInterval # = [] (by default)
    finalAlpha # = 0 (by default)
    finalBeta # = 0 (by default)
    reRun # = false (by default)
    root # = [] (by default)
    function TrackedInterval(interval)
        ndim = Int(length(interval)/2)
        new(interval,interval,[],ndim,false,false,false,[],false,fill(type(0.0394555475981047),ndim),[],[],[],[],[], 1, 0, false,[])
    end
end

"""==============================FUNCTIONS FOR TRACKED INTERVAL=============================="""

function canThrowOut(trackedInterval::TrackedInterval)
    """Ensures that an interval that has not subdivided cannot be thrown out on the final step."""
    return !trackedInterval.finalStep || trackedInterval.canThrowOutFinalStep
end

function addTransform(trackedInterval::TrackedInterval, subInterval)
    """Adds the next alpha and beta values to the list transforms and updates the current interval.

    Parameters:
    -----------
    subInterval : array
        The subinterval to which the current interval is being reduced
    """
    #Ensure the interval has non zero size; mark it empty if it doesn't
    if any(subInterval[1,:] > subInterval[2,:]) && canThrowOut(trackedInterval)
        trackedInterval.empty = true
        return
    elseif any(subInterval[1,:] > subInterval[2,:])
        #If we can't throw the interval out, it should be bounded by [-1,1].
        subInterval[1,:] = min.(subInterval[1,:], ones(type,length(subInterval[1,:])))
        subInterval[1,:] = max.(subInterval[1,:], -ones(type,length(subInterval[1,:])))
        subInterval[2,:] = min.(subInterval[2,:], ones(type,length(subInterval[1,:])))
        subInterval[2,:] = max.(subInterval[2,:], subInterval[1,:])
    end
    # Get the alpha and beta associated with the transformation in each dimension
    a1 = subInterval[1,:]
    b1 = subInterval[2,:] # all the lower bounds and upper bounds of the new interval, respectively
    a2 = trackedInterval.interval[1,:]
    b2 = trackedInterval.interval[2,:] # all the lower bounds and upper bounds of the original interval
    alpha1, beta1 = (b1-a1)/type(2), (b1+a1)/type(2)
    alpha2, beta2 = (b2-a2)/type(2), (b2+a2)/type(2)
    push!(trackedInterval.transforms,hcat(alpha1, beta1))
    #Update the lower and upper bounds of the current interval
    for dim in 0:trackedInterval.ndim-1
        for i in 0:1
            x = subInterval[i+1,dim+1]
            #Be exact if x = +-1
            if x == -1.0
                trackedInterval.interval[i+1,dim+1] = trackedInterval.interval[1,dim+1]
            elseif x == 1.0
                trackedInterval.interval[i+1,dim+1] = trackedInterval.interval[2,dim+1]
            else
                trackedInterval.interval[i+1,dim+1] = alpha2[dim+1]*x+beta2[dim+1]
            end
        end
    end
end

function getLastTransform(trackedInterval::TrackedInterval)
    return trackedInterval.transforms[end]
end

function getFinalInterval(trackedInterval::TrackedInterval)
    """Finds the point that should be reported as the root (midpoint of the final step interval).

    Returns
    -------
    root: numpy array
        The final point to be reported as the root of the interval
    """
    finalInterval = trackedInterval.topInterval'
    finalIntervalError = zeros(type,size(finalInterval))
    transformsToUse = trackedInterval.finalStep ? trackedInterval.preFinalTransforms : trackedInterval.transforms
    for transform in reverse(transformsToUse) # Iteratively apply each saved transform
        alpha = transform[:,1]
        beta = transform[:,2]
        finalInterval, temp = twoProd(finalInterval, alpha)
        finalIntervalError = alpha .* finalIntervalError + temp
        finalInterval, temp = twoSum(finalInterval,beta)
        finalIntervalError += temp
    end
    finalInterval = finalInterval'
    finalIntervalError = finalIntervalError'
    trackedInterval.finalInterval = finalInterval + finalIntervalError # Add the error and save the result.
    trackedInterval.finalAlpha, alphaError = twoSum(-finalInterval[1,:] ./ type(2), finalInterval[2,:] ./ type(2))
    trackedInterval.finalAlpha += alphaError + (finalIntervalError[2,:] - finalIntervalError[1,:]) ./ type(2)
    trackedInterval.finalBeta, betaError = twoSum(finalInterval[1,:] ./ type(2), finalInterval[2,:] ./ type(2))
    trackedInterval.finalBeta += betaError + (finalIntervalError[2,:] + finalIntervalError[1,:]) ./ type(2)
    return trackedInterval.finalInterval
end

function getFinalPoint(trackedInterval::TrackedInterval)
    """Finds the point that should be reported as the root (midpoint of the final step interval).

    Returns
    -------
    root: numpy array
        The final point to be reported as the root of the interval
    """
    if !trackedInterval.finalStep  # If no final step, use the midpoint of the calculated final interval.
        trackedInterval.root = (trackedInterval.finalInterval[1,:] .+ trackedInterval.finalInterval[2,:]) ./ type(2)
    else  # If using the final step, recalculate the final interval using post-final transforms.
        finalInterval = trackedInterval.topInterval'
        finalIntervalError = zeros(type,size(finalInterval))
        transformsToUse = trackedInterval.transforms
        for transform in reverse(transformsToUse)
            alpha = transform[:,1]
            beta = transform[:,2]
            finalInterval, temp = twoProd(finalInterval, alpha)
            finalIntervalError = alpha .* finalIntervalError + temp
            finalInterval, temp = twoSum(finalInterval, beta)
            finalIntervalError += temp
        end
        finalInterval = finalInterval' .+ finalIntervalError'
        trackedInterval.root = (finalInterval[1,:] .+ finalInterval[2,:]) ./ type(2)  # Return the midpoint
    end
    return trackedInterval.root
end

# not thoroughly tested
function sizeOfInterval(trackedInterval)
    """Gets the volume of the current interval."""
    return prod(trackedInterval.interval[2,:] - trackedInterval.interval[1,:])
end

function dimSize(trackedInterval)
    """Gets the lengths along each dimension of the current interval."""
    return trackedInterval.interval[2,:] - trackedInterval.interval[1,:]
end

function finalDimSize(trackedInterval)
    """Gets the lengths along each dimension of the current interval."""
    return trackedInterval.finalInterval[2,:] - trackedInterval.finalInterval[1,:]
end

function copyInterval(trackedInterval::TrackedInterval)
    """Returns a deep copy of the current interval with all changes and properties preserved."""
    newone = TrackedInterval(trackedInterval.topInterval)
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

function contains(trackedInterval::TrackedInterval, point)
    """Determines if point is contained in the current interval."""
    # Elementwise, not `>=`/`<=`. Those compare vectors lexicographically in Julia, so
    # they stop at the first differing coordinate: with x in [-1,1] and y in [-2,2],
    # the point [0, 5] compared lexicographically is "less than" [1, 2] on its first
    # coordinate alone and was reported as contained.
    return all(point .>= trackedInterval.interval[1,:]) && all(point .<= trackedInterval.interval[2,:])
end

function overlapsWith(trackedInterval::TrackedInterval, otherInterval::TrackedInterval)
    """Determines if the otherInterval overlaps with the current interval.

    Returns True if the lower bound of one interval is less than the upper bound of the other
        in EVERY dimension; returns False otherwise."""
    currentInterval = getIntervalForCombining(trackedInterval)
    otherInterval = getIntervalForCombining(otherInterval)
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

function isPoint(trackedInterval::TrackedInterval, macheps = type(2)^-(precision-1))
    """Determines if the current interval has essentially length 0 in each dimension."""
    return all(abs.(trackedInterval.interval[1,:] - trackedInterval.interval[2,:]) .< macheps)
end

function startFinalStep(trackedInterval::TrackedInterval)
    """Prepares for the final step by saving the current interval and its transform list."""
    trackedInterval.finalStep = true
    trackedInterval.preFinalInterval = copy(trackedInterval.interval)
    trackedInterval.preFinalTransforms = copy(trackedInterval.transforms)
end

function getIntervalForCombining(trackedInterval::TrackedInterval)
    """Returns the interval to be used in combining intervals to report at the end."""
    return trackedInterval.finalStep ? trackedInterval.preFinalInterval : trackedInterval.interval
end

function toStr(trackedInterval::TrackedInterval)
    return string(trackedInterval.interval)
end
