# YRoots.jl

[![CI](https://github.com/wlgns0330/Julia-Rootfinding/actions/workflows/CI.yml/badge.svg)](https://github.com/wlgns0330/Julia-Rootfinding/actions/workflows/CI.yml)
[![Docs](https://github.com/wlgns0330/Julia-Rootfinding/actions/workflows/docs.yml/badge.svg)](https://github.com/wlgns0330/Julia-Rootfinding/actions/workflows/docs.yml)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://wlgns0330.github.io/Julia-Rootfinding)

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

- [ChebSolve dependency chart](https://viewer.diagrams.net/?highlight=0000ff&edit=_blank&nav=1&title=ChebSolve.drawio#R7V1bV%2Bo8E%2F413nxryWrS86WKVLcnFF%2F3qzeuUgJUS4NtqeDF%2Fu1fCgRKE6BbSw%2B8eCMNbWkzM88cMjM5Es8GY8Mzh%2F0b3EHOERQ64yOxfgQhVHWV%2FItGJrMREarSbKTn2Z3ZmLAcaNlfaDYI6OjI7iB%2FPjYbCjB2Anu4Omhh10VWsDJmeh7%2BXD2ti53OysDQ7KGVx4gGWpbpIOa033Yn6M9GNTl29gWye336y0CYfzMw6cnzAb9vdvBnbEg8PxLPPIyD2afB%2BAw50eytzktjzbeLB%2FOQG6S54PP2TH57v5Ivrju9l3MB3uvHF8cAzm4Tms5o%2Fsbzpw0mdApC5AU2mZFrs42cJvbtwMYu%2BaqNgwAPjsRTesKJY%2FeiLwI8JKP9YOCQA0A%2BklcfRjcbjHsRm9Tapm9bNW9Kr9Ou7Thn2MEemZG6i10UXRB4%2BH0x4dNbzDiDvKN4arud2TvL5GD%2B%2FOQR0HjtzIDFfBNORXiAAm9CTplfIGmzK%2BY8egxUoNTk2djnkujkxDnv9lcoTnncnLNab%2FEDS2qQD3OC%2FAVx1OJoY2HPRd6rh0du53VOqKlUS0JNJZ8%2F%2B3aAWkPTii79JBfGCTl9NLHROK%2Bf17OhEBA0nVKEUkkHOodIRAA1lkgi0HdEI3E7jXhztZ0ADh4tZn6XMyuJiWmFOjOpClTZOYUa%2BPmc9lpf6Bfy%2Fnl4fvv9cjE4FYnQHwPOnCrOdCIweav45CofI0y%2FOJ5hxAk5AWjD8XR%2B6PfkUy%2F6%2F%2BiZrt%2FF3uCsj9qXbtMhVAH1c6IovIaHEP0d8tizn5pdxZA0Rj5Mpr%2FrTIG97WDrnQxN%2F0cEt93e%2FCyuvJBfmCs8gjeRVPmLa3RyNBU%2B1JnfwWz72BkF6MSz6EXR6OJIZ2Dz1rwlY87IsjuX0V06g9CQred7%2BfmPvngphlM4%2FLSJeWBSLDWVh50iUFgOoribOQOxSo1P97zoOh2Z%2FwrYOZ3f3dCwtTP90cZiRnQGIKEiVU3k6EcO8krKjmjMAu8W2W4NHTvYD0m%2BJBQOr%2Fu1%2FsOJlZUkQyUpyaoqsHIsckygLGj82LAR%2BlA%2BLKLmrvyTrtD4%2FX5MGSpGL9Qh5vn8EHtBH%2Fewazrny9HYZEe24vKcaxyZPtO5fkNBMJnPtjkK8KpCRmM7%2BDe6vAbl%2BeHz%2FG7R5%2Fo4fjChBy554dlVMj18pjeMDpaXTY%2FodS6OPfuMDRdPPB9YS14fjzwLbZKRuSoPTK%2BHgk0nzikYze5GdvGQYwZ2uOoc8eg%2Bv7SJ7anGpmymAcaO01Stpsf%2B5NWbzp59fp8EPy0e7PswQnm8KBb7Foep5WExJSWHqVkz2I%2BUh7LWwmxTm88y3ce%2Bhz%2FvRkHMKmwXaxLO1ELMFRi5PqIewl10ahBRPQpCfEfrJPwMS%2ByqlrZVG7VHoeGhp9pDO4qtZKGNdFFNqCKB4%2BlJME9VJBeJE%2FTz8wpmbMOJOEjE1FIRMKGlVURyqXBC244T5H2uTT9YGJ8HsNgIFi4BC2mkDH%2B1ThvZgIUkCqtgAYDCGq5A5Tmgu3JOVLHiaFEkVqQ1Win%2BlwQr9O1YMQ1QHfBhEz4MCT48gl%2B3b43Pf7LBB0WVki4HkGgkPx7k3JFnywcI6eB2%2FAQj6PLgdoNCLxVI0OfehBIde9A6AMU2oPggQHF97xz%2FEhobWelvYmBycolJUxmY0PO0I8D6pZAFT3RtIlD1A8%2Bk5Rn4b0sYXY0z4hlV0hjtokCFtT956kXOYOWYr16UCqoXWCL1AlOql3KFtVS1NG7HN5yOYkPlaSmefaj8ZxqCl4CQ0BCvr7ZrB6%2BvB%2B2wUTv4RDscB%2FBmgp%2BzsigUZVU1aILKWRynSx65mBRKMRhBJR6URuKBlFLiSxZnoM8dk3jy%2FI3ICrx0A%2BSR8YNQrwh1KN0%2BESsNZSXUmlQTBABUTQIQChpMmH9A5iamSXlmR6iA4YHyG4Blii%2FIFQUHeS04LJfQD8gQ3T0gyOAe6ydyMM5K3ctqMpNY0DlIIIIckQCkWOt%2BfbUIBU3b9Q9G4hauUT5D41IcyKPmvZIR1%2Bgqk0TJDyGoOss3OwshaBVf8C5WfagVVR9syUEk34459H%2FbhA8POBDHgabz1L4fy%2FfZ4ICqJUFAVCBHeQg5Kg%2Bt0DjiKgioFUx7ARXNewEpEl9sf25RHuyFDTihEpz45y5s4iv4kA1OSEy5GpTYfBeNYyvsDiYKztNO722mkPbvZcT8ACOKy3dZk6S9YJSlJhITpTo7zsqmc7IxmYY8w8yzbQVoeIChjTB0Og6NP5p0cvfv6CkbGFKShYNA1jnmipRnbFsrNKtm1VyBFTRXYNqsGlCudU%2Bq7MpA981k%2F77%2B2coZOZCdinxJyA5TJFORF6KrIg3sneFB23YjrD6oi03q4hdRFzqSGp1LNyPvVkqWdBCbglUXcFflhfyM3UKztL%2B5SFKeNZJFP5ZtsAGlcsFGuqS6ztO0dQby%2F%2Fe%2F8qBF1pXE10TQr84VyzT8jHoCKEKydguwSZQ7q93iy3llFkN3K6%2BwmvJK3Yryk6%2BcMJ2W7GLJrDuYyrq7tl1keo%2FIG%2Fj7C9M3BKYvvaD1R%2FzIqCoGiokYD4FpXtaKnGfWil7FqpgSpS1DsZoID1MksTpTOT%2FrI%2Bsd7K%2Bgf36FxkPXtJ6fejdZOV6JON0aQRdzFfQK1ieUR8rTJq6WbFERsomr7YikRJpoiGamzFsTP0CD%2FRDoMRHo0O6cBsZ1PyOBVhLrfxqvFxdNHchFnCGbdRh84tYoNxrm22BtQmiqdAVJ6PZOjqJ6zahN46zvJhHeICM6A1lSajCRE6Isc8XiBQTibojN7YQCio2axZE7bWJYeYoOYNp%2BSaBc0M0mkRIBb3q4E6WF7VFnvad6aJyY%2FwQvAf7KSIw1ETAtz2h%2BzUqB6I6ygvlCXHDbM7XqQZW0CZ6gME%2BLT%2FdiQ6F%2FT%2FQSQXdqkpfM7Oa1EZ%2F5zwRL3RVmWGmn%2FDkHoqihsou9QTSHnJbKc0UQ88lnt83XLc%2FXApyqCbl2b2lnGakJKMiJNlYaTfePG3p5dl8ttJi8%2BmH3tKm%2FsLBcGi7Vi03lrLh1n5rm5WoiAFOke%2FszI39fA6%2B%2FCaLbjSY4GdffskJ0RWRaanMqOhTO3gq7i9SkSKwlrLvoZjnN8N3jhbXueWi0wWnXg%2BOLrPw9VY%2FcmxW6AxmwyhwKnFIeagVkb%2FkXvLb2PY%2BvRItrYuqWc%2BWK3YjpsiQjkY8k3owimdNNE%2FZX6odE6n3roWE1v0bZSL2qJju9cJLhIcg1TlvBNbYS2e9UireLOyyV%2FQ4qU4tVTrLDtGQvV5RHr6C3Hid6vPtLIXRPmzpTMrqLbOqMhzojC7WwE6JO3d6TdfQWUdnotlnzJsJHRv5ZsspR03mrMjtq98rfZKKCQlwqEz11Zky54i8imxlj9VF74YXnt1PdboU4aITG%2BaQnGleGl5G3DZhqeEHRePltuwqzcAW5itr4LwX5B1Ja0d5qIpvllNm%2BpF8YDy7dO5cmwl2Sf4VtRiruHAgaBAhGrRvk1IU%2F2QCBTuNby64YgsjBAW7QLYuNSLk4IFYxob1E%2BjxttlTZkIJNlyLP3xq1O3Zo%2BzZeiLm%2FH3r9goizpYzv7q8VLSO9DpNlhACKnDRXIU%2BtTkNCRUnzd7YjLc%2FiqJi6F1bJjPN0m8BduoS7fXTndVY0955Fzi%2BJpL81vsDpXdfISNIlIam5gQQ49vuOYud8fcKWiPtRPCXaWXri91EYw%2FL9IGxohMY97jWwc42ycs1UnclfBzLdWiOO4lKeaU1SFXf8LY9NRk2t7TZZuZZB6XMnRbqJnckDskaeHz3CXsjyhMjy2JNezoJmRrFSAEEyzMJtTqrtqDkp3xwrdKULxMW4eh2KpbTrHSVzrSR2vcP2z8fEnbKxV%2Bz%2BF7sQ4zc0%2FDX0xIyipSqTlyQKnK1y5R31DuVLcXn2rKreVrlSRfewkdgFD8anCjx7cLPHqYcnRLyH%2FosFjF47G%2FFW5KSSlgU27VDLs9BfYmPmy93I8qHi5sqQfNvvdZEqqMJW1oAXUbsXNHz6%2BJ3VJhNQlllnTKF0j5eYwDzZg61NsvBwcmCNtazx8YsYBVYbXfTGVxkZBRpg85UVic2FUHIFDl7UbT1XxLU90FIwgLiY6SQlV%2BdfqGksdZdUhZG%2B32quTFtVn3jelGtd7KKpldChI5SPyVDDdjZHDI6FmjB9%2BGigSaxeMt3IYyyUmiyIK74GuUqiA8nLltxlXoeGfflxK8P%2B%2Fa5tl7QOSNowQtyC5S3yLQZzMnF4lRb%2FLQbeEL6KtZolLDcZXhn39Z6YjuU2WdvfKST7PgvDtJkMaTvrZc%2FD899ItviX9URXKIXuT01vMnv5nXX4l3lNWHciHjEdK5oSkMzyCw4X%2BTeJ1CqK925D4%2FleOx65EYmyEKmsBUdKnQJUUfSX2Wjzgb23sveShTt3oSGd%2BW8vWB2kZOGM4zY%2FYO7MN3%2FLTS1AJdGEIPedX2ReK8j%2Fltm03u6nA5vs%2FvWW1hORqX67q76c9pSfq4UfWE1pxaOi0M%2BLbf43OZg1wrlpSbPvYrUe96HRHR2fNFD4p5z2S9rawKqyMK9l0MF62ZbIsSmVfsnddis0XrTa6y1QHkrK3alr4crm10JJrqmSvviTk%2B0zNWkR68zJoNEL3Uf5e30XOOi8iQcTkexGQ1QF%2BWcmQtp1zZI1MmdV%2F6KGK0odvHSbDgHF2%2FomNK3OCuaDGxpnzucprp9n1TwFkAdPNstUOVUcNBU7l7UIVhuanc6CsMXQMt%2FFqZTbhr2PQkO%2BGbq3t484G4bQhSQ3CJzdhMRcqwAK3WwwF0D%2FPnCnzw5NG%2FvIKcmfjdcFceR%2BxPuVW3Z5HhoP%2FvV5TWyK2UiqpmlJ5AaCovNK8ECeC8mApSyTajTLOShLolEpod0j0A5uLqD1%2FmhlxDAw0X9BFVheUWGOyK4c3N6j7W4vq3%2BWfq%2B6we%2FVHkPjU%2FHUtz%2BGXU6%2FN23cPnWDt7zcXoUmbVFRkpWEk7tmNZezAsDs0yrQm%2BfkMNPZXaeK90MBfxEF%2FOejF95dOL2MTOVkep%2BucnK7cy210ni6t2q2c6k63KRucKNnbV6vwQsxCT0aSHQwTQ09zK10Ld%2B1Rzppa8v26%2FZgTyr2ozpeV3cvel0yV9mEbkRm7VhQOA21tB1FbvgAVLCvXvVaz7Rok3kUdp11QvdEYHns7%2BGGKWkEmeU6kEMPRw1%2BlqcTOe%2Ff4A6Kzvg%2F)
  — how the pieces of the solver fit together, rendered from `ChebSolve.drawio` in this
  repository.
- [YRoots: A Robust Numerical Rootfinder](https://arxiv.org/abs/2401.02114) — the paper
  behind the algorithm.
- [The parent yroots project](https://wlgns0330.github.io/RootFinding/)
