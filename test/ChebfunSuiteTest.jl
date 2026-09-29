import YRoots
using Test

# The Chebfun2 benchmark systems, ported from tests/test_chebfun2_suite.py in
# wlgns0330/RootFinding.
#
# Each case is a 2-D polynomial or transcendental system with a known root set. The
# reference roots in ChebfunTests/Polished_results/ were obtained by polishing
# high-accuracy solutions of these systems (they are the Python suite's .npy files,
# written out as CSV), and are treated here as ground truth. The systems themselves are
# the ones in ChebfunTests/ChebfunVars.jl.
#
# Every case is checked three ways, so a failure says which property broke:
#   - the number of roots: no missed roots, no spurious ones;
#   - every root makes the system vanish (residual check; independent of the reference);
#   - the roots are in the right places (agreement with the reference; independent of
#     the residuals). A root set can have tiny residuals and still be wrong when the
#     system is ill conditioned near a root.
#
# DEFAULT_RESIDUAL_TOL and DEFAULT_ROOT_TOL cover most cases. A system that needs a
# looser bound sets it in CHEBFUN_TOLERANCES with the measured value beside it, and every
# bound leaves at least an order of magnitude over that measurement.
#
# ChebfunTests/Chebfun_results/ holds the roots Chebfun2 itself reports. Nothing here
# asserts against them: Chebfun is a second implementation, not ground truth.

module ChebfunCases
include(joinpath(@__DIR__, "ChebfunTests", "ChebfunVars.jl"))
end

const CHEBFUN_POLISHED_DIR = joinpath(@__DIR__, "ChebfunTests", "Polished_results")

"""Largest |f(root)| tolerated, for systems that do not override it."""
const DEFAULT_RESIDUAL_TOL = 1e-12
"""Largest distance (max norm) tolerated between a found root and its reference root."""
const DEFAULT_ROOT_TOL = 1e-12

# Per-case overrides, as (residual, root) tolerances, with the value measured on each.
const CHEBFUN_TOLERANCES = Dict(
    # Products of near-tangent curves: the roots are badly conditioned even though the
    # residuals stay at 1e-14. Measured root error 1.4e-11.
    "1.2" => (DEFAULT_RESIDUAL_TOL, 1e-9),
    # Nearly tangent circles. Measured root error 7.7e-12.
    "3.1" => (DEFAULT_RESIDUAL_TOL, 1e-10),
    # Same tangency problem as 3.1, with 45 roots. Measured root error 3.7e-13.
    "3.2" => (DEFAULT_RESIDUAL_TOL, 1e-11),
    # Degree 18 and 10 with coefficients up to 1e8, so the residual is large in absolute
    # terms while the roots are accurate. Measured residual 1.1e-7, root error 8.7e-14.
    "4.2" => (1e-6, 1e-11),
    # 6.1 with the double root split apart by 1e-4. Measured root error 3.9e-13.
    "6.2" => (DEFAULT_RESIDUAL_TOL, 1e-11),
    # Degree 10 in a rescaled variable, poorly conditioned near the roots. Measured
    # residual 5.5e-12, root error 8.3e-9.
    "7.2" => (1e-10, 1e-7),
    # The search box is 1e-9 across, so the default absolute root tolerance would be
    # meaningless. Measured root error 6.2e-25.
    "7.3" => (DEFAULT_RESIDUAL_TOL, 1e-22),
)

# Cases whose root count is known to be wrong. Their count and one-to-one match are
# @test_broken, which fails once they pass, so this list cannot go stale.
const CHEBFUN_BROKEN_COUNT = Dict(
    # The double root at the origin comes out of the final step as two possible
    # duplicates, 6 roots for 5. Whether they merge depends on last-bit rounding; the
    # Python solver reports 5 on the same system.
    "6.1" => "double root reported twice",
)

"""The reference roots of case `name`, as a vector of 2-vectors."""
function chebfunReference(name)
    return [parse.(Float64, split(line, ",")) for line in eachline(joinpath(CHEBFUN_POLISHED_DIR, "polished_$name.csv"))]
end

"""
Max-norm distance between found and reference roots, and whether each found root's
nearest reference root is a different one. Two found roots collapsing onto the same
reference root is not one-to-one, which a nearest-neighbour distance alone would hide.
"""
function chebfunMatch(found, reference)
    isempty(found) && return 0.0, isempty(reference)
    nearest = [argmin([maximum(abs.(r .- p)) for p in reference]) for r in found]
    distance = maximum(maximum(abs.(r .- reference[k])) for (r, k) in zip(found, nearest))
    return distance, allunique(nearest) && length(found) == length(reference)
end

function test_all_ChebfunSuite()
    @testset "Chebfun2 benchmark systems" begin
        for (k, fileName) in enumerate(ChebfunCases.file_names)
            name = replace(fileName, "yroots_roots_" => "")
            f, g = ChebfunCases.fs[k], ChebfunCases.gs[k]
            residualTol, rootTol = get(CHEBFUN_TOLERANCES, name, (DEFAULT_RESIDUAL_TOL, DEFAULT_ROOT_TOL))
            @testset "case $name" begin
                found = YRoots.solve([f, g], ChebfunCases.as[k], ChebfunCases.bs[k])
                reference = chebfunReference(name)
                distance, oneToOne = chebfunMatch(found, reference)
                if haskey(CHEBFUN_BROKEN_COUNT, name)
                    @test_broken length(found) == length(reference)
                    @test_broken oneToOne
                else
                    @test length(found) == length(reference)
                    @test oneToOne
                end
                residual = isempty(found) ? 0.0 : maximum(max(abs(f(r...)), abs(g(r...))) for r in found)
                @test residual <= residualTol
                @test distance <= rootTol
            end
        end
    end
end
