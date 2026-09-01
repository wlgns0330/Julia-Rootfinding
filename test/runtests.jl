using Pkg
using Test

# Activate the project environment. Resolved from this file's location, so the
# script works from any working directory (and from CI).
Pkg.activate(dirname(@__DIR__))
Pkg.instantiate()

println("Testing...")
include("ChebyshevApproximatorTest.jl")
include("ChebyshevSubdivisionSolverTest.jl")
include("TrackedIntervalTest.jl")
include("QuadraticCheckTest.jl")
include("../../Julia-Rootfinding/src/StructsWithTheirFunctions/TrackedInterval.jl")
include("../../Julia-Rootfinding/src/StructsWithTheirFunctions/SolverOptions.jl")
include("PolynomialTests.jl")
include("SolveApiTest.jl")

# Each suite runs inside one outer testset so that a failure in one does not abort the
# rest -- an inner testset that fails records the failure and the run continues, where
# calling them bare let the first failing suite hide every suite after it.
function test_all()
    @testset "YRoots" begin
        test_all_SolveApi()
        test_all_ChebyshevApproximator()
        test_all_ChebyshevSubdivisionSolver()
        test_all_TrackedInterval()
        test_all_QuadraticCheck()
        test_all_Polynomial()
    end
end

# Running this file runs the whole suite (this is what CI does).
# To run a subset locally, comment this out and uncomment one of the lines below.
test_all()

# Uncomment the lines below to run specific test sets
# 
# ============================================ All Tests ============================================
# test_all()
#
# ============================================ ChebyshevApproximator Tests ============================================
# test_all_ChebyshevApproximator()
# 
# test_transformPoints()
# test_getFinalDegree()
# test_startedConverging()
# test_checkConstantInDimension()
# test_hasConverged()
# test_createMeshgrid()
# test_getApproxError()
# test_intervalApproximateND()
# test_getChebyshevDegrees()
# test_chebApproximate()
# 
# ============================================ ChebyshevSubdivisionSolver Tests ============================================
# test_all_ChebyshevSubdivisionSolver()
# 
# test_getLinearTerms()
# test_linearCheck1()
# test_reduceSolveDim()
# test_transformChebInPlace1D()
# test_transformChebInPlaceND()
# test_getTransformationError()
# test_transformCheb()
# test_transformChebToInterval()
# test_getSubdivisionDims()
# test_getInverseOrder()
# test_getSubdivisionIntervals()
# test_boundingIntervalLinearSystem()
# test_zoomInOnIntervalIter()
# test_isExteriorInterval()
# test_trimMs()
# test_solvePolyRecursive()
# test_solveChebyshevSubdivision()

# ============================================ TrackedInterval Tests ============================================

# test_all_TrackedInterval()

# test_copyInterval()
# test_addTransform()
# test_getIntervalForCombining()
# test_isPoint()
# test_getFinalInterval()
# test_getFinalPoint()
# test_overlapsWith()
# test_startFinalStep()

# ============================================ QuadraticCheck Tests ============================================

# test_all_QuadraticCheck()


# test_quadraticCheck2D()
# test_quadraticCheck3D()
# test_get_fixed_vars()
# test_quadraticCheckND()
# test_quadraticCheck()

# ============================================ PolynomialTests Tests ============================================

# test_all_Polynomial()

# test_construction()
# test_eval_MultiPower()