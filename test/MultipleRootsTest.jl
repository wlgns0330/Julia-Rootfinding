import YRoots
using LinearAlgebra: qr
using Test

# Systems with multiple (non-simple) roots, in one through four dimensions.
#
# Ported from tests/test_multiple_roots.py in the Python solver. A multiple root can only be
# located to about sqrt(macheps) (or macheps^(1/m) for multiplicity m), and the solver may report
# it as a small cluster of points, with a warning listing them as possible duplicates. So these
# tests check that every true root has a reported root nearby and a bounding box around it, and
# that nothing is reported away from the true roots. They do not check the number of points
# reported or the warning.
#
# The systems come in both sparse and dense flavors. Polynomials such as (x-.3)^2 have sparse
# Chebyshev coefficient tensors, while compositions with sin, exp, tanh or cos have dense ones.

"""Solve the system, dropping the duplicate root warnings multiple roots produce."""
function solveQuietly(funcs, a, b; kwargs...)
    _, result = Test.collect_test_logs(min_level=Base.CoreLogging.Warn) do
        YRoots.solve(funcs, a, b; kwargs...)
    end
    return result
end

"""
Test that every expected root is reported and boxed, and that every reported root is an expected one.

`boxSlack` widens the bounding boxes for expected roots that are themselves only known to within
rounding error. `checkBoxes=false` skips the bounding box check, for a case where whether the box
contains the root depends on rounding (see ROUNDING_DEPENDENT_BOX).
"""
function testFindsExactly(funcs, a, b, expected; tol=1e-6, boxSlack=0.0, checkBoxes=true)
    roots, boxes = solveQuietly(funcs, a, b; returnBoundingBoxes=true)
    @test length(roots) > 0
    isempty(roots) && return
    for root in expected
        @test minimum(maximum(abs.(r .- root)) for r in roots) < tol
        checkBoxes && @test any(all(box[1, :] .- boxSlack .<= root .<= box[2, :] .+ boxSlack) for box in boxes)
    end
    for r in roots
        @test minimum(maximum(abs.(r .- root)) for root in expected) < tol
    end
end

"""The root of the increasing function `f` on `[lo, hi]`, by bisection."""
function bisect(f, lo, hi)
    for _ in 1:200
        mid = (lo + hi) / 2
        (mid == lo || mid == hi) && break
        f(mid) < 0 ? (lo = mid) : (hi = mid)
    end
    return (lo + hi) / 2
end

# chebApproximate's error bound covers only the truncated tail of the series, not rounding in the
# coefficients (5.1e-16 for (x-.3)^3 against a stated 3.5e-21). The approximation can then sit
# farther from the function than its stated error, and whether the bounding box still contains the
# true root depends on the last bits of the arithmetic: here the 1D triple root's box, about 1e-12
# wide, ends about 2e-9 short of it, while dropping either of the rounding pads in
# boundingIntervalLinearSystem moves it back over the root. The Python tests mark the same case as
# a non-strict xfail. Its root is still checked; only its box is not.
const ROUNDING_DEPENDENT_BOX = Set(["1D triple root"])

const ONE_MULTIPLE_ROOT = [
    # sparse
    ("1D triple root", [x -> (x - 0.3)^3], [-1.0], [1.0], [[0.3]]),
    ("2D fourfold root, both functions touch zero",
        [(x, y) -> (x - 0.2)^2, (x, y) -> (y + 0.1)^2], [-1.0, -1.0], [1.0, 1.0], [[0.2, -0.1]]),
    ("2D fourfold root, both functions touch zero along diagonals",
        [(x, y) -> (x + y - 0.1)^2, (x, y) -> (x - y - 0.3)^2], [-1.0, -1.0], [1.0, 1.0], [[0.2, -0.1]]),
    ("2D fourfold root, one function touches zero only at the root",
        [(x, y) -> (x - 0.2)^2 + (y + 0.1)^2, (x, y) -> (x - 0.2) * (y + 0.1)], [-1.0, -1.0], [1.0, 1.0],
        [[0.2, -0.1]]),
    ("3D double root",
        [(x, y, z) -> (x - 0.3)^2, (x, y, z) -> y - 0.1, (x, y, z) -> z + 0.2], fill(-1.0, 3), fill(1.0, 3),
        [[0.3, 0.1, -0.2]]),
    ("3D eightfold root",
        [(x, y, z) -> (x - 0.2)^2, (x, y, z) -> (y + 0.1)^2, (x, y, z) -> (z - 0.3)^2], fill(-1.0, 3),
        fill(1.0, 3), [[0.2, -0.1, 0.3]]),
    ("4D double root",
        [(a, b, c, d) -> (a - 0.3)^2, (a, b, c, d) -> b - 0.1, (a, b, c, d) -> c + 0.2, (a, b, c, d) -> d - 0.4],
        fill(-1.0, 4), fill(1.0, 4), [[0.3, 0.1, -0.2, 0.4]]),
    # dense: sin(...)^2 touches zero along a curve that the other functions cross once
    ("2D dense double root",
        [(x, y) -> sin(x + 2y - 0.5)^2, (x, y) -> exp(x - y^2 - 0.1) - 1], [-1.0, -1.0], [1.0, 1.0],
        [[(sqrt(1.4) - 1)^2 + 0.1, sqrt(1.4) - 1]]),
    ("3D dense double root",
        [(x, y, z) -> sin(x + y + z - 0.3)^2, (x, y, z) -> exp(x) - exp(y + 0.1), (x, y, z) -> sin(z - 0.2x)],
        fill(-1.0, 3), fill(1.0, 3), [[0.4 / 2.2, 0.4 / 2.2 - 0.1, 0.2 * 0.4 / 2.2]]),
    ("4D dense double root",
        [(a, b, c, d) -> sin(a + b + c + d - 0.3)^2, (a, b, c, d) -> exp(a) - exp(b + 0.1),
         (a, b, c, d) -> sin(c - 0.2a), (a, b, c, d) -> tanh(d + 0.3b)],
        fill(-1.0, 4), fill(1.0, 4), [[0.37 / 1.9, 0.37 / 1.9 - 0.1, 0.2 * 0.37 / 1.9, -0.3 * (0.37 / 1.9 - 0.1)]]),
]

"""sin(pi*(x+y))^2 touches zero on the lines x + y = k; x - y = .3cos(x) crosses each once."""
function dense2DRoots()
    roots = Vector{Float64}[]
    for k in (-1, 0, 1)
        x = bisect(x -> 2x - k - 0.3cos(x), -2.0, 2.0)
        push!(roots, [x, k - x])
    end
    return roots
end

const SEVERAL_MULTIPLE_ROOTS = [
    # sparse
    ("1D two double roots", [x -> (x + 0.5)^2 * (x - 0.4)^2], [-1.0], [1.0], [[-0.5], [0.4]]),
    ("2D two double roots",
        [(x, y) -> ((x - 0.3) * (x + 0.4))^2, (x, y) -> y - 0.1], [-1.0, -1.0], [1.0, 1.0],
        [[0.3, 0.1], [-0.4, 0.1]]),
    # Neither function touches zero here: the circle and the ellipse are tangent at two points.
    ("2D circle tangent to an ellipse at two points",
        [(x, y) -> (x - 0.1)^2 + (y - 0.05)^2 - 0.25, (x, y) -> (x - 0.1)^2 / 0.25 + (y - 0.05)^2 / 0.04 - 1],
        [-1.0, -1.0], [1.0, 1.0], [[0.6, 0.05], [-0.4, 0.05]]),
    ("3D two double roots",
        [(x, y, z) -> ((x - 0.3) * (x + 0.4))^2, (x, y, z) -> y - 0.1x, (x, y, z) -> z + y^2 - 0.2],
        fill(-1.0, 3), fill(1.0, 3), [[0.3, 0.03, 0.2 - 0.03^2], [-0.4, -0.04, 0.2 - 0.04^2]]),
    # dense
    ("1D three double roots", [x -> sin(5x)^2], [-1.0], [1.0], [[-pi / 5], [0.0], [pi / 5]]),
    ("2D three double roots",
        [(x, y) -> sin(pi * (x + y))^2, (x, y) -> x - y - 0.3cos(x)], [-1.0, -1.0], [1.0, 1.0], dense2DRoots()),
]

"""A fixed orthogonal `dim x dim` matrix, from the QR factorization of a matrix with entries in [-1, 1]."""
orthogonalMatrix(dim, seed) = Matrix(qr([sin(seed * (7i + 3j)) for i in 1:dim, j in 1:dim]).Q)

"""
The "devastating example" of Noferini and Townsend, p_i(x) = x_i^2 + eps*(Qx)_i.

See V. Noferini and A. Townsend, "Numerical instability of resultant methods for multidimensional
rootfinding", SIAM J. Numer. Anal. 54 (2016). Q is orthogonal. The root at the origin is simple for
eps > 0 but, as eps -> 0, merges with other roots into a root of multiplicity 2^dim, which makes
resultant methods lose accuracy exponentially in dim.
"""
devastating(Q, eps) = [(x...) -> x[i]^2 + eps * sum(Q[i, j] * x[j] for j in 1:size(Q, 1)) for i in 1:size(Q, 1)]

function test_all_MultipleRoots()
    @testset "All tests in MultipleRootsTest.jl" begin
        # The fourfold and eightfold roots, where every function touches zero, zoom to a box about
        # 1e-7 wide whose linear terms are all below the approximation error. isBelowResolution
        # stops them there rather than subdividing without end.
        @testset "a single multiple root is found: $name" for (name, funcs, a, b, expected) in ONE_MULTIPLE_ROOT
            testFindsExactly(funcs, a, b, expected; checkBoxes=!(name in ROUNDING_DEPENDENT_BOX))
        end
        @testset "several multiple roots are all found: $name" for (name, funcs, a, b, expected) in SEVERAL_MULTIPLE_ROOTS
            testFindsExactly(funcs, a, b, expected)
        end
        @testset "the devastating example with eps = 0 in $(dim)D has one root of multiplicity 2^dim" for dim in 2:4
            testFindsExactly(devastating(orthogonalMatrix(dim, 1), 0.0), fill(-1.0, dim), fill(1.0, dim), [zeros(dim)])
        end
        # Substituting x = eps*u turns the system into u_i^2 + (Qu)_i = 0, which does not depend on eps.
        # So the roots are eps times the roots of the eps = 1 system, whose roots are simple and well
        # separated. Summing u_i^2 = -(Qu)_i over i gives |u|^2 <= sqrt(dim)|u|, so all of them lie in
        # the box [-sqrt(dim), sqrt(dim)]^dim.
        #
        # The accuracy of the small roots is absolute, about sqrt(macheps), so their relative accuracy
        # degrades as eps shrinks; that is expected for a near-multiple root. The tolerance is kept below
        # eps so that the check still tells the cluster's roots apart from the origin.
        @testset "the devastating example in $(dim)D at eps = $eps finds the whole cluster" for dim in 2:4, eps in (1e-2, 1e-4, 1e-6)
            Q = orthogonalMatrix(dim, 1)
            bound = sqrt(dim) + 0.5
            unitRoots = solveQuietly(devastating(Q, 1.0), fill(-bound, dim), fill(bound, dim))
            atOrigin = [maximum(abs.(r)) < 1e-12 for r in unitRoots]
            @test any(atOrigin)
            @test length(unitRoots) > 1
            expected = [o ? zeros(dim) : eps .* r for (r, o) in zip(unitRoots, atOrigin)]
            testFindsExactly(devastating(Q, eps), fill(-1.0, dim), fill(1.0, dim), expected;
                             tol=min(1e-6, eps / 4), boxSlack=1e-13 * eps)
        end
    end
end
