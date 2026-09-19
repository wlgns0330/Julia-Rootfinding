# YRoots.jl

[![CI](https://github.com/wlgns0330/Julia-Rootfinding/actions/workflows/CI.yml/badge.svg)](https://github.com/wlgns0330/Julia-Rootfinding/actions/workflows/CI.yml)
[![Docs](https://github.com/wlgns0330/Julia-Rootfinding/actions/workflows/docs.yml/badge.svg)](https://github.com/wlgns0330/Julia-Rootfinding/actions/workflows/docs.yml)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://wlgns0330.github.io/Julia-Rootfinding)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wlgns0330/Julia-Rootfinding/blob/main/CombinedNotebook.ipynb)

Find all the roots of a system of smooth multivariate functions on a compact interval.

This is a Julia refactor of the [YRoots](https://arxiv.org/abs/2401.02114) Python package.
Given a list of smooth functions and a search box, YRoots builds a Chebyshev approximation
of each function on that box, uses properties of those approximations to shrink the box,
and subdivides and recurses where they cannot shrink it further — returning one point, and
optionally a bounding box, per root.

📖 **[Full documentation](https://wlgns0330.github.io/Julia-Rootfinding)**

## Installation

The package is not registered, so install it from this repository. Julia 1.11 or newer is
required.

```julia
using Pkg
Pkg.add(url = "https://github.com/wlgns0330/Julia-Rootfinding")
```

## Quick start

```julia
using YRoots

# Roots of x + y = 0.3 and x - y = 0.1 on [-1,1]^2
solve([(x, y) -> x + y - 0.3,
       (x, y) -> x - y - 0.1], [-1.0, -1.0], [1.0, 1.0])
# 1-element Vector{Any}:
#  [0.2, 0.09999999999999994]
```

`solve(funcs, a, b)` takes a vector of functions, a vector of lower bounds and a vector of
upper bounds — one entry per dimension, in dimension order. One function per dimension is
expected. The roots come back in no particular order:

```julia
solve([x -> sin(10x)], [-1.0], [1.0])
# 7-element Vector{Any}:
#  [-0.9424777960769379]
#  [-0.6283185307179587]
#  [-0.31415926535897953]
#  [-4.067093094220837e-17]
#  [0.6283185307179585]
#  [0.942477796076938]
#  [0.3141592653589792]
```

Ask for bounding boxes to see how tightly each root was localised. Each box is a `2 x dim`
array whose first row holds the lower bound in every dimension and whose second row holds
the upper bound:

```julia
roots, boxes = solve([(x, y) -> x + y - 0.3,
                      (x, y) -> x - y - 0.1], [-1.0, -1.0], [1.0, 1.0];
                     returnBoundingBoxes = true)
# boxes[1] == [0.19999999999999937 0.0999999999999993
#              0.2000000000000007  0.10000000000000063]
```

### Polynomial input

A function is approximated before it is solved. If you already have a polynomial, hand its
coefficient tensor to `MultiPower` (monomial basis) or `MultiCheb` (Chebyshev basis) and
the approximation step is skipped:

```julia
M1 = MultiPower([0 3 0 2; 1.5 0 7 0; 0 0 4 -2; 0 0 0 1])
M2 = MultiCheb([0.02 0.31; -0.43 0.19; 0.06 0])

solve([M1, M2], [-5.0, -5.0], [5.0, 5.0])
# 1-element Vector{Any}:
#  [-0.34050318985851813, 0.17101211857064774]
```

Both types also evaluate directly, via `eval_MultiPower` and `eval_MultiCheb`.

### Keyword arguments

| Keyword | Default | Effect |
| --- | --- | --- |
| `verbose` | `false` | Print approximation shapes and rootfinding progress to the terminal. |
| `returnBoundingBoxes` | `false` | Return `(roots, boundingBoxes)` instead of just the roots. |
| `exact` | `false` | Run the transformations in higher precision — more accurate, slower. |
| `minBoundingIntervalSize` | `1e-5` | Re-solve on a smaller interval while a root's box exceeds this size. |
| `roundoff` | `53` | Bits of precision. `53` selects `Float64` and the fast solver; `<= 24` `Float32`, `<= 11` `Float16`, above `53` `BigFloat`. |

See the [`solve` reference](https://wlgns0330.github.io/Julia-Rootfinding/solve/) for the
details.

### Things to know

- **The input has to be well behaved.** `solve` is only guaranteed to work well when every
  function is continuous and smooth on the search interval and every root in it is simple.
  A non-smooth function, or an interval holding infinitely many roots, drives the solver
  into deep subdivision; it gives up at the depth cap and reports a wide bounding box.
- **The first solve of a new dimension is slow.** The solver specialises on the dimension
  of the system. Dimensions 1 through 5 are precompiled when the package is installed, so
  their first solve returns promptly; a higher dimension pays a one-off compilation cost of
  several seconds on its first call.

## Repository layout

```
src/
  YRoots.jl                     module entry point: load order, exports, precompile workload
  CombinedSolver.jl             solve() — approximate, then solve, then refine
  ChebyshevApproximator.jl      Chebyshev approximation of a function on an interval
  ChebyshevSubdivisionSolver.jl the subdivision solver over Chebyshev coefficients
  QuadraticCheck.jl             exclusion test that discards root-free subintervals
  StructsWithTheirFunctions/    SolverOptions, TrackedInterval, MultiCheb/MultiPower
  FastSolve/                    the Float64 fast path, used by default (roundoff = 53)
test/
  runtests.jl                   runs every suite; what CI runs
  ChebfunTests/                 accuracy comparison against Chebfun's published results
docs/                           Documenter site published to GitHub Pages
```

## Development

Run the test suite from the repository root:

```bash
julia --project=. test/runtests.jl
```

`test/runtests.jl` is a plain script rather than a `Pkg.test` entrypoint, and calls
`test_all()` at the bottom. To run one suite while working on it, comment that call out and
uncomment one of the individual test functions listed below it.

Build the documentation locally:

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path = pwd())); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The site lands in `docs/build`. Pushes to `main` publish it to GitHub Pages automatically.

### Contributing

Use the [Forking Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow):
fork the repository, commit to a branch on your fork, and open a pull request against
`main`. CI runs the test suite on Julia 1.11 and on the current stable release; please make
sure it passes before asking for a review.

## Reference

- [YRoots: A Robust Numerical Rootfinder](https://arxiv.org/abs/2401.02114) — the paper
  behind the algorithm.
- [The parent yroots project](https://wlgns0330.github.io/RootFinding/)
