import YRoots
using Test

# Tests for the public entry point, YRoots.solve(funcs, a, b).
#
# Ported from tests/test_solve_api.py, tests/test_bounding_boxes.py and
# tests/test_known_failures.py in wlgns0330/Rootfinding-serial, keeping the cases that
# apply to this port. Everything here goes through the exported `solve`, so it exercises
# whichever path the default `roundoff` selects (currently fast_solve) rather than the
# internals -- the internals are covered by the per-module test files.
#
# The functions are given as callables throughout, deliberately: MultiPower and MultiCheb
# use different coefficient layouts from their Python counterparts, so a system written
# out as raw coefficients would be asserting a layout convention rather than the solver.
# Polynomial input is covered in PolynomialTests.jl.
#
# NOT ported, because the Julia solve() has no equivalent behavior to assert:
#   - rejects non-callable input, inverted bounds, mismatched bound lengths, and a
#     polynomial whose dimension is not the system size. Julia performs no input
#     validation; each of these either throws an unrelated MethodError deep in the
#     approximator or silently returns nonsense.
#   - a bare function outside a list, and scalar bounds broadcast to every dimension.
#     `dim = length(funcs)` requires an indexable container, and the bounds are used as
#     vectors, so neither form is accepted.
# See test_solve_api.py for what those would look like if the validation is added.

const SOLVE_ATOL = 1e-8

"""Sort a solve() result into a canonical order so two runs can be compared elementwise."""
function sortedRoots(roots)
    return sort([Float64.(r) for r in roots], by = Tuple)
end

"""Largest absolute residual of `f` over `roots`."""
function maxResidual(f, roots)
    isempty(roots) && return 0.0
    return maximum(abs(f(r...)) for r in roots)
end

function test_all_SolveApi()
    @testset "All tests in SolveApiTest.jl" begin
        test_solve_findsKnownRoots()
        test_solve_searchBox()
        test_solve_illConditioned()
        test_solve_noRoots()
        test_solve_boundingBoxes()
        test_solve_options()
        test_solve_accuracy()
        test_solve_doesNotModifyInputs()
    end
end

function test_solve_findsKnownRoots()
    @testset "solve finds known roots" begin
        # A linear system has exactly one root, at the obvious place.
        roots = YRoots.solve([(x, y) -> x - 0.5, (x, y) -> y + 0.25], [-1.0, -1.0], [1.0, 1.0])
        @test length(roots) == 1
        @test length(roots[1]) == 2
        @test isapprox(roots[1], [0.5, -0.25], atol = SOLVE_ATOL)

        # sin(pi x) = 0 and y = x^2  ->  (-1,1), (0,0), (1,1)
        f = (x, y) -> sin(pi * x)
        g = (x, y) -> y - x^2
        roots = sortedRoots(YRoots.solve([f, g], [-1.5, -0.5], [1.5, 2.5]))
        @test length(roots) == 3
        for (got, want) in zip(roots, [[-1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
            @test isapprox(got, want, atol = SOLVE_ATOL)
        end

        # Three dimensions, chained so each variable is pinned by the previous one.
        roots = YRoots.solve([(x, y, z) -> x - 0.5,
                              (x, y, z) -> y + x,
                              (x, y, z) -> z - y^2],
                             [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
        @test length(roots) == 1
        @test isapprox(roots[1], [0.5, -0.5, 0.25], atol = SOLVE_ATOL)

        # One dimension, where the solver has no subdivision axis to choose between.
        roots = sortedRoots(YRoots.solve([x -> x^2 - 0.25], [-1.0], [1.0]))
        @test length(roots) == 2
        @test isapprox(roots[1], [-0.5], atol = SOLVE_ATOL)
        @test isapprox(roots[2], [0.5], atol = SOLVE_ATOL)
    end
end

function test_solve_searchBox()
    @testset "solve respects the search box" begin
        # Roots well outside [-1,1] are found once the box contains them.
        roots = YRoots.solve([(x, y) -> x - 3.5, (x, y) -> y + 2.25], [-5.0, -5.0], [5.0, 5.0])
        @test length(roots) == 1
        @test isapprox(roots[1], [3.5, -2.25], atol = 1e-7)

        # A box that is not centered on the origin. cos(x) = 0.5 at x = pi/3 in [0,2].
        roots = YRoots.solve([(x, y) -> cos(x) - 0.5, (x, y) -> y - x], [0.0, 0.0], [2.0, 2.0])
        @test length(roots) == 1
        @test isapprox(roots[1], [pi / 3, pi / 3], atol = SOLVE_ATOL)

        # Every root reported must lie inside the box that was searched.
        a, b = [-0.9, -0.9], [0.9, 0.9]
        roots = YRoots.solve([(x, y) -> sin(3 * x * y), (x, y) -> y - x^3], a, b)
        @test length(roots) > 0
        for r in roots
            @test all(a .<= r) && all(r .<= b)
        end
    end
end

function test_solve_illConditioned()
    @testset "solve keeps the root of an ill conditioned system" begin
        # x + y = 0.3 and x + (1+eps) y = 0.3 are two nearly parallel lines meeting at
        # (0.3, 0). The interval padding that keeps rounding error from discarding a root
        # is derived from the reciprocal condition number, so as the system stiffens the
        # padding shrinks toward machine precision and the root can fall outside it.
        illConditioned(eps) = ((x, y) -> x + y - 0.3, (x, y) -> x + (1 + eps) * y - 0.3)

        # 1e-4, 1e-6 and 1e-7 used to return no roots at all here. widthToAdd was
        # computed as max(S[end]/S[1], 2) * machEps, but S[end]/S[1] is the reciprocal
        # condition number and never exceeds 1, so the max was always exactly 2 and the
        # padding stayed at machine epsilon however stiff the system was.
        #
        # NOT covered below: eps = 1e-10, where solve() does not terminate. That is a
        # separate, pre-existing defect -- unmodified main hangs on it identically -- and
        # it is left out deliberately rather than asserted, because a hanging test takes
        # CI down on a timeout instead of reporting a failure. invCondNum there is ~1e-10,
        # right on the `wellConditioned` threshold, so the padding lands near 2e-6 and the
        # interval can no longer shrink below it; the zoom loop keeps reporting a change
        # and subdivides without end.
        for eps in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9)
            f, g = illConditioned(eps)
            roots = YRoots.solve([f, g], [-1.0, -1.0], [1.0, 1.0])
            @test length(roots) == 1
            if length(roots) == 1
                @test isapprox(roots[1], [0.3, 0.0], atol = 1e-6)
            end
        end
    end
end

function test_solve_noRoots()
    @testset "solve on a system with no roots" begin
        # x^2 + y^2 + 3 and x + y + 9 are both bounded away from zero on [-1,1]^2.
        funcs = [(x, y) -> x^2 + y^2 + 3, (x, y) -> x + y + 9]
        roots = YRoots.solve(funcs, [-1.0, -1.0], [1.0, 1.0])
        @test length(roots) == 0

        roots, boxes = YRoots.solve(funcs, [-1.0, -1.0], [1.0, 1.0]; returnBoundingBoxes = true)
        @test length(roots) == 0
        @test length(boxes) == 0
    end
end

function test_solve_boundingBoxes()
    @testset "solve bounding boxes" begin
        f = (x, y) -> sin(4 * (x + y / 10 + pi / 10))
        g = (x, y) -> cos(2 * (x - 2 * y + pi / 7))
        a, b = [-1.0, -1.0], [1.0, 1.0]
        roots, boxes = YRoots.solve([f, g], a, b; returnBoundingBoxes = true)

        @test length(roots) > 0
        @test length(boxes) == length(roots)
        for (root, box) in zip(roots, boxes)
            # Boxes come back as 2 x dim: row 1 is the lower bound in each dimension,
            # row 2 the upper. (The Python solver returns the transpose of this.)
            @test size(box) == (2, length(root))
            lo, hi = box[1, :], box[2, :]
            # The box brackets its own root, sits inside the search interval, and is tight.
            @test all(lo .<= root) && all(root .<= hi)
            @test all(lo .>= a .- 1e-12) && all(hi .<= b .+ 1e-12)
            @test all(hi .- lo .< 1e-4)
        end

        # Asking for the boxes must not change which roots come back.
        funcs = [(x, y) -> x - 0.5, (x, y) -> y + 0.25]
        plain = sortedRoots(YRoots.solve(funcs, [-1.0, -1.0], [1.0, 1.0]))
        withBoxes, _ = YRoots.solve(funcs, [-1.0, -1.0], [1.0, 1.0]; returnBoundingBoxes = true)
        withBoxes = sortedRoots(withBoxes)
        @test length(plain) == length(withBoxes)
        for (p, w) in zip(plain, withBoxes)
            @test isapprox(p, w, atol = SOLVE_ATOL)
        end
    end
end

function test_solve_options()
    @testset "solve keyword options" begin
        f = (x, y) -> sin(4 * (x + y / 10 + pi / 10))
        g = (x, y) -> cos(2 * (x - 2 * y + pi / 7))
        a, b = [-1.0, -1.0], [1.0, 1.0]

        # exact=true runs the transformations in higher precision; it must not change
        # which roots are found, only how tightly they are pinned down.
        fast = sortedRoots(YRoots.solve([f, g], a, b; exact = false))
        exact = sortedRoots(YRoots.solve([f, g], a, b; exact = true))
        @test length(fast) == length(exact)
        for (p, q) in zip(fast, exact)
            @test isapprox(p, q, atol = 1e-8)
        end

        # A larger minBoundingIntervalSize stops the re-solve loop sooner, so the boxes
        # it returns cannot be tighter than the default's.
        _, tight = YRoots.solve([f, g], a, b; returnBoundingBoxes = true)
        _, loose = YRoots.solve([f, g], a, b; returnBoundingBoxes = true,
                                minBoundingIntervalSize = 1e-2)
        @test length(tight) == length(loose)
        widest(boxes) = maximum(maximum(box[2, :] .- box[1, :]) for box in boxes)
        @test widest(tight) <= widest(loose) + 1e-12
    end
end

function test_solve_accuracy()
    @testset "solve accuracy" begin
        # Every reported root must actually satisfy the system it came from.
        f = (x, y) -> exp(x + y) - 2
        g = (x, y) -> x - y^2
        roots = YRoots.solve([f, g], [-1.0, -1.0], [1.0, 1.0])
        @test length(roots) > 0
        @test maxResidual(f, roots) < 1e-10
        @test maxResidual(g, roots) < 1e-10

        # No root may be reported twice. A system with several well separated roots
        # exercises the duplicate-combining logic in the final step.
        f = (x, y) -> sin(4 * (x + y / 10 + pi / 10))
        g = (x, y) -> cos(2 * (x - 2 * y + pi / 7))
        roots = YRoots.solve([f, g], [-1.0, -1.0], [1.0, 1.0])
        @test length(roots) > 1
        for i in 1:length(roots), j in (i + 1):length(roots)
            @test maximum(abs.(roots[i] .- roots[j])) > 1e-6
        end
    end
end

function test_solve_doesNotModifyInputs()
    @testset "solve does not modify its inputs" begin
        a, b = [-2.0, -2.0], [2.0, 2.0]
        aCopy, bCopy = copy(a), copy(b)
        funcs = [(x, y) -> x - 0.5, (x, y) -> y + 0.25]
        YRoots.solve(funcs, a, b)
        @test a == aCopy
        @test b == bCopy
    end
end
