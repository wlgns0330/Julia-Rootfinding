module YRoots

include("CombinedSolver.jl")

export solve

# Compile the solver at each dimension, once at install, instead of on each session's first solve.
#
# Solving is specialized on the dimension of the system, so without this the first solve of a given
# dimension compiles the whole solver before returning any roots -- around 26s for the first 2-D
# system, 9s for the first 3-D and 29s for the first 4-D -- against a few milliseconds for every
# solve after it. Running systems here moves that into precompilation, which Julia caches to disk.
#
# A `precompile(solve, ...)` declaration is not enough: the dimension comes from `length(funcs)`, a
# runtime value, so nothing downstream of it can be inferred from the signature alone. Measured, a
# declaration alone still leaves 8-10s on each first solve.
#
# Each dimension costs precompile time (about 55s total for 1 through 4); raise the range to cover
# higher dimensions.
let
    for n in 1:4
        eqs = Any[(x...) -> x[i] - x[i+1] for i in 1:n-1]
        push!(eqs, (x...) -> sum(v^2 for v in x) - 0.5)
        solve(eqs, fill(-1.0, n), fill(1.0, n))
    end
end

end # module YRoots
