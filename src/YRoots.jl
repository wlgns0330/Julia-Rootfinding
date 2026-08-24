module YRoots #This is the main module file, for versioning and test suite compatibility

include("CombinedSolver.jl")

export solve

# Solving is specialized on the dimension of the system, so on a fresh session the first solve of
# each dimension pays for compiling the whole solver at that dimension before it returns any
# roots -- around 26s for the first 2-D system and another 9s for the first 3-D one, against a
# few milliseconds for every solve after it. Running small systems here moves that compilation
# into precompilation, where it is paid once when the package is installed (or when Julia or a
# dependency changes) and then cached to disk for every session afterwards.
#
# Each dimension listed costs precompile time, so this covers 1-D through 3-D; a first 4-D solve
# still compiles on demand.
let
    solve(Any[x -> x^2 - 0.25], [-1.0], [1.0])
    solve(Any[(x, y) -> x^2 + y^2 - 0.5, (x, y) -> x - y], [-1.0, -1.0], [1.0, 1.0])
    solve(Any[(x, y, z) -> x^2 + y^2 - 0.5,
              (x, y, z) -> x - y,
              (x, y, z) -> z - 0.25], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
    solve(Any[(w, x, y, z) -> w^2 + x^2 - 0.5,
              (w, x, y, z) -> w - x,
              (w, x, y, z) -> y - 0.25,
              (w, x, y, z) -> z + 0.25], [-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0])
end

end # module YRoots
