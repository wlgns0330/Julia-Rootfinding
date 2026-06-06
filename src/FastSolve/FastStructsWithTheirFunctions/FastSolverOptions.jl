mutable struct FastSolverOptions
    """Settings for running interval checks, transformations, and subdivision in solvePolyRecursive.

    Parameters
    ----------
    verbose : bool
        Defaults to false. Whether or not to output progress of solving to the terminal.
    exact : bool
        Defaults to false. Whether the transformation in TransformChebInPlaceND should minimize error.
    constant_check : bool
        Defaults to true. Whether or not to run constant term check after each subdivision.
    low_dim_quadratic_check : bool
        Defaults to true. Whether or not to run quadratic check in dim 2, 3.
    all_dim_quadratic_check : bool
        Defaults to false. Whether or not to run quadratic check in dim >= 4.
    maxZoomCount : int
        Maximum number of zooms allowed before subdividing (prevents infinite infintesimal shrinking)
    level : int
        Depth of subdivision for the given interval.
    """
    verbose::Bool                  # = false (by default)
    exact::Bool                    # = false (by default)
    constant_check::Bool           # = true (by default)
    low_dim_quadratic_check::Bool  # = true (by default)
    all_dim_quadratic_check::Bool  # = true (by default)
    maxZoomCount::Int              # = 25 (by default)
    level::Int                     # = 0 (by default)
    useFinalStep::Bool             # = true (by default)
    
    function FastSolverOptions(verbose=false,exact=false,constant_check=true,low_dim_quadratic_check=true,all_dim_quadratic_check=true,maxZoomCount=25,level=0,useFinalStep=true)
        #Init all the Options to default value
        new(verbose,exact,constant_check,low_dim_quadratic_check,all_dim_quadratic_check,maxZoomCount,level,useFinalStep)
    end
end

function fast_copySO(SO)
    return FastSolverOptions(SO.verbose,SO.exact,SO.constant_check,SO.low_dim_quadratic_check,SO.all_dim_quadratic_check,SO.maxZoomCount,SO.level,SO.useFinalStep)
end