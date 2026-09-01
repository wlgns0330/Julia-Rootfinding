include("../src/QuadraticCheck.jl")
using Test

# Tests for QuadraticCheck.jl, reworked from tests/test_QuadraticCheck.py in
# wlgns0330/Rootfinding-serial.
#
# The quadratic check is one sided: returning true is a guarantee that the polynomial,
# together with its error bound, has no root in [-1,1]^n, while returning false means
# nothing at all. So the property worth testing is soundness -- whenever the check
# certifies, the polynomial really is bounded away from zero -- not agreement with a
# recorded answer. These tests assert that directly, by evaluating the polynomial on a
# grid whenever the check certifies it.
#
# This replaces a file of hardcoded expected values transcribed from the Python
# implementation. Those pinned the answers but not the guarantee: they could not tell a
# correct certification from an unsound one, and said nothing about the boundary between
# the two.
#
# COEFFICIENT LAYOUT. Julia stores the tensor with its axes reversed relative to Python:
# for 2-D, coeff[i, j] multiplies T_{j-1}(x) * T_{i-1}(y), so the x degree is the *column*
# index. quadraticCheck2D reads the linear x term from test_coeff[1,2] and the linear y
# term from test_coeff[2,1], and takes `shape = reverse(size(test_coeff))`. Anything
# ported from the Python tests therefore has its axes reversed on the way in.
# (For the grid soundness checks the layout happens not to matter -- [-1,1]^n is symmetric
# under permuting coordinates, so the transpose takes the same set of values -- but the
# fixtures below are written in the real layout so they mean what they say.)

const QC_TOL = 1e-8

# ------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------

"""T_n(x) by the Chebyshev recurrence."""
function chebT(n::Int, x::Float64)
    n == 0 && return 1.0
    n == 1 && return x
    tm2, tm1 = 1.0, x
    for _ in 2:n
        tm2, tm1 = tm1, 2x * tm1 - tm2
    end
    return tm1
end

"""
Evaluate a Chebyshev tensor in the solver's reversed layout at `pt`.

`coeff[i_d, ..., i_1]` multiplies `T_{i_1-1}(pt[1]) * ... * T_{i_d-1}(pt[d])`.
"""
function chebvalND(coeff, pt)
    d = ndims(coeff)
    total = 0.0
    for I in CartesianIndices(coeff)
        c = coeff[I]
        c == 0 && continue
        term = Float64(c)
        for k in 1:d
            term *= chebT(I[d - k + 1] - 1, Float64(pt[k]))
        end
        total += term
    end
    return total
end

"""Smallest |f| seen on an `n`-per-axis grid over [-1,1]^d. An upper bound on the true min."""
function gridMinAbs(coeff, n::Int)
    d = ndims(coeff)
    grid = range(-1.0, 1.0; length = n)
    best = Inf
    for idx in CartesianIndices(ntuple(_ -> n, d))
        v = abs(chebvalND(coeff, ntuple(k -> grid[idx[k]], d)))
        v < best && (best = v)
    end
    return best
end

# A small deterministic generator, so the randomized sweeps are reproducible without
# taking a dependency on Random (which is not in the project's [deps]).
mutable struct QCRng
    state::UInt64
end
QCRng(seed::Integer) = QCRng(UInt64(seed) * 6364136223846793005 + 1442695040888963407)

function nextUnit!(rng::QCRng)
    rng.state = rng.state * 6364136223846793005 + 1442695040888963407
    return Float64(rng.state >> 11) / Float64(1 << 53)
end

"""Approximately standard normal, by the central limit theorem on 12 uniforms."""
nextNormal!(rng::QCRng) = sum(nextUnit!(rng) for _ in 1:12) - 6.0

nextChoice!(rng::QCRng, options) = options[1 + floor(Int, nextUnit!(rng) * length(options))]

# ------------------------------------------------------------------------------------

function test_all_QuadraticCheck()
    @testset "All tests in QuadraticCheckTest.jl" begin
        test_get_fixed_vars()
        test_quadraticCheck_dispatch()
        test_quadraticCheck_certifies()
        test_quadraticCheck_soundness()
        test_quadraticCheck_randomSoundness()
        test_quadraticCheck_agreement()
        test_quadraticCheck_tolerance()
        test_quadraticCheck_inputHandling()
    end
end

function test_get_fixed_vars()
    @testset "get_fixed_vars unit tests" begin
        # Julia indices, largest subsets first.
        @test get_fixed_vars(1) == []
        @test get_fixed_vars(2) == [(1,), (2,)]
        @test get_fixed_vars(3) == [(1, 2), (1, 3), (2, 3), (1,), (2,), (3,)]

        # The structural property behind those literals: every subset fixes at least one
        # variable and leaves at least one free, so neither the corners (all fixed) nor
        # the interior (none fixed) are ever produced, and none is repeated.
        for dim in 2:5
            subsets = get_fixed_vars(dim)
            @test all(0 < length(s) < dim for s in subsets)
            @test length(subsets) == length(unique(subsets))
            @test all(all(1 .<= collect(s) .<= dim) for s in subsets)
            # every proper nonempty subset of 1:dim, so 2^dim - 2 of them
            @test length(subsets) == 2^dim - 2
        end
    end
end

function test_quadraticCheck_dispatch()
    @testset "quadraticCheck dispatches on dimension" begin
        coeff2 = zeros(3, 3)
        coeff2[1, 1] = 10.0
        @test quadraticCheck(coeff2, QC_TOL) == quadraticCheck2D(coeff2, QC_TOL)

        coeff3 = zeros(3, 3, 3)
        coeff3[1, 1, 1] = 10.0
        @test quadraticCheck(coeff3, QC_TOL) == quadraticCheck3D(coeff3, QC_TOL)

        # nd_check=true routes even a 2-D tensor through the general implementation.
        @test quadraticCheck(coeff2, QC_TOL, true) == quadraticCheckND(copy(coeff2), QC_TOL)

        # The dimension specific checks refuse anything of the wrong rank rather than
        # indexing off the end of it.
        @test quadraticCheck2D(ones(3, 3, 3), QC_TOL) == false
        @test quadraticCheck3D(ones(3, 3), QC_TOL) == false

        # One dimensional input has no quadratic form to check; it must still answer.
        @test quadraticCheck([5.0, 0.1, 0.1], QC_TOL) isa Bool
        @test quadraticCheck([0.0, 1.0, 0.0], QC_TOL) == false   # T_1 = x, root at 0
    end
end

function test_quadraticCheck_certifies()
    @testset "quadraticCheck certifies dominant constants" begin
        # A constant term that dominates everything else cannot vanish on the box. Each
        # case is cross checked against the polynomial's actual minimum on a grid, so the
        # fixture cannot silently stop meaning what it says.
        coeff = zeros(3, 3)
        coeff[1, 1], coeff[2, 2], coeff[1, 3] = 10.0, 0.5, 0.3
        @test quadraticCheck2D(coeff, QC_TOL)
        @test gridMinAbs(coeff, 60) > QC_TOL

        coeff = zeros(3, 3, 3)
        coeff[1, 1, 1], coeff[1, 1, 2], coeff[2, 2, 1] = 10.0, 1.0, 0.5
        coeff[1, 1, 3], coeff[2, 2, 2] = 0.4, 0.05
        @test quadraticCheck3D(coeff, QC_TOL)
        @test gridMinAbs(coeff, 21) > QC_TOL

        coeff = zeros(3, 3, 3, 3)
        coeff[1, 1, 1, 1], coeff[1, 1, 1, 2], coeff[2, 2, 1, 1] = 8.0, 0.5, 0.25
        @test quadraticCheckND(coeff, QC_TOL)
    end
end

function test_quadraticCheck_soundness()
    @testset "quadraticCheck refuses polynomials with roots" begin
        # f = x, a root along the whole line x = 0.
        x = zeros(3, 3)
        x[1, 2] = 1.0
        @test quadraticCheck2D(x, QC_TOL) == false

        # x^2 + y^2 - 1.25 in the Chebyshev basis: T_2 = 2t^2 - 1, so 0.5*T_2(x) +
        # 0.5*T_2(y) - 0.25 = x^2 + y^2 - 1.25, which vanishes on a circle through the box.
        circle = zeros(3, 3)
        circle[1, 1], circle[1, 3], circle[3, 1] = -0.25, 0.5, 0.5
        @test quadraticCheck2D(circle, QC_TOL) == false

        # f = xy, a saddle vanishing on both axes.
        saddle = zeros(3, 3)
        saddle[2, 2] = 1.0
        @test quadraticCheck2D(saddle, QC_TOL) == false

        # f = z, and a sphere, in three dimensions.
        z = zeros(3, 3, 3)
        z[2, 1, 1] = 1.0
        @test quadraticCheck3D(z, QC_TOL) == false

        sphere = zeros(3, 3, 3)
        sphere[1, 1, 1] = -0.5
        sphere[1, 1, 3] = sphere[1, 3, 1] = sphere[3, 1, 1] = 0.5
        @test quadraticCheck3D(sphere, QC_TOL) == false

        # A constant smaller than the tolerance is not bounded away from zero as far as
        # the caller is concerned, so it must not be certified.
        small = zeros(3, 3)
        small[1, 1] = 1e-12
        @test quadraticCheck2D(small, QC_TOL) == false
    end
end

function test_quadraticCheck_randomSoundness()
    @testset "quadraticCheck certifications are sound" begin
        # The central property: a certification is a promise that |f| > tol on the whole
        # box. Sweep a spread of coefficient tensors and, every time the check certifies
        # one, verify the promise on a grid.
        certified2D = 0
        for seed in 1:6
            rng = QCRng(seed)
            for _ in 1:50
                scale = nextChoice!(rng, (1.0, 0.3, 3.0))
                coeff = [nextNormal!(rng) * scale for _ in 1:3, _ in 1:3]
                coeff[1, 1] += nextChoice!(rng, (0.0, 3.0, -3.0))
                if quadraticCheck2D(coeff, QC_TOL)
                    certified2D += 1
                    @test gridMinAbs(coeff, 40) > QC_TOL
                end
            end
        end
        # The sweep has to actually reach the certifying branch, or it proves nothing.
        @test certified2D > 0

        certified3D = 0
        for seed in 1:3
            rng = QCRng(100 + seed)
            for _ in 1:40
                coeff = zeros(3, 3, 3)
                for _ in 1:6
                    i = 1 + floor(Int, nextUnit!(rng) * 3)
                    j = 1 + floor(Int, nextUnit!(rng) * 3)
                    k = 1 + floor(Int, nextUnit!(rng) * 3)
                    coeff[i, j, k] = nextNormal!(rng) * 0.4
                end
                coeff[1, 1, 1] = nextChoice!(rng, (0.0, 3.0, -3.0)) + 0.2 * nextNormal!(rng)
                if quadraticCheck3D(coeff, QC_TOL)
                    certified3D += 1
                    @test gridMinAbs(coeff, 15) > QC_TOL
                end
            end
        end
        @test certified3D > 0
    end
end

function test_quadraticCheck_agreement()
    @testset "quadraticCheck 2D/3D and ND agree" begin
        # The dimension specific checks are hand specialized versions of the general one.
        # They must reach the same verdict on the same input.
        for seed in 1:4
            rng = QCRng(200 + seed)
            for _ in 1:40
                coeff = [nextNormal!(rng) for _ in 1:3, _ in 1:3]
                coeff[1, 1] += nextChoice!(rng, (0.0, 3.0))
                @test quadraticCheck2D(coeff, QC_TOL) == quadraticCheckND(copy(coeff), QC_TOL)
            end
        end

        for seed in 1:3
            rng = QCRng(300 + seed)
            for _ in 1:25
                coeff = zeros(3, 3, 3)
                for _ in 1:6
                    i = 1 + floor(Int, nextUnit!(rng) * 3)
                    j = 1 + floor(Int, nextUnit!(rng) * 3)
                    k = 1 + floor(Int, nextUnit!(rng) * 3)
                    coeff[i, j, k] = nextNormal!(rng) * 0.4
                end
                coeff[1, 1, 1] = nextChoice!(rng, (0.0, 3.0, -3.0))
                @test quadraticCheck3D(coeff, QC_TOL) == quadraticCheckND(copy(coeff), QC_TOL)
            end
        end
    end
end

function test_quadraticCheck_tolerance()
    @testset "quadraticCheck is monotone in the tolerance" begin
        # tol is added to the slack the check has to overcome, so raising it can only make
        # certification harder. Anything certified at a large tolerance must also be
        # certified at a small one.
        rng = QCRng(400)
        for _ in 1:60
            coeff = [nextNormal!(rng) for _ in 1:3, _ in 1:3]
            coeff[1, 1] += 3.0
            if quadraticCheck2D(coeff, 1.0)
                @test quadraticCheck2D(coeff, 1e-12)
            end
        end
    end
end

function test_quadraticCheck_inputHandling()
    @testset "quadraticCheck input handling" begin
        # The checks are predicates and must not write to the tensor they are handed.
        for coeff in (ones(3, 3), ones(3, 3, 3), ones(3, 3, 3, 3))
            original = copy(coeff)
            quadraticCheck(coeff, QC_TOL)
            @test coeff == original
            quadraticCheckND(coeff, QC_TOL)
            @test coeff == original
        end

        # Tensors smaller than the quadratic part: the guarded reads in quadraticCheck2D
        # must not index off the end.
        small = [5.0 0.1; 0.1 0.0]          # only degree <= 1 terms present
        @test quadraticCheck2D(small, QC_TOL)
        @test quadraticCheckND(copy(small), QC_TOL)

        tiny = fill(7.0, 1, 1)
        @test quadraticCheck2D(tiny, QC_TOL)
        @test quadraticCheckND(copy(tiny), QC_TOL)

        # Tensors larger than the quadratic part: everything above degree 2 is folded into
        # the slack term, so a small high order tail is still certifiable...
        coeff = zeros(5, 5)
        coeff[1, 1], coeff[5, 5] = 10.0, 0.5
        @test quadraticCheck2D(coeff, QC_TOL)
        @test quadraticCheckND(copy(coeff), QC_TOL)

        # ...and a large one is not, because it can dominate the constant.
        coeff[5, 5] = 20.0
        @test quadraticCheck2D(coeff, QC_TOL) == false
    end
end
