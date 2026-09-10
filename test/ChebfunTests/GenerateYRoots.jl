using Pkg
include("ChebfunVars.jl")
include(joinpath(@__DIR__, "..", "..", "src", "CombinedSolver.jl"))

using DelimitedFiles

# Define a function to compare two vectors element by element. returns true if v1 < v2 
function compare_vectors(v1, v2; tolerance=1e-9)
    for (a, b) in zip(v1, v2)
        if abs(a - b) < tolerance #If the values are essentially equal
            continue
        end
        return a < b
    end
    return false  # If all elements are equal within tolerance, consider them equal
end

# fs and gs are the input functions
# as are the lower bounds of the interval, bs are the upper bounds.
# file_names are the names for the output CSV files.
for (f, g, a, b, file_name) in zip(fs, gs, as, bs, file_names)
    roots = solve([f, g], a, b) # returns a vector of vectors of floats

    # Sort the roots using the custom comparison function
    roots = sort!(roots, by = x -> x, lt = (v1, v2) -> compare_vectors(v1, v2))
    
    # Save the results to a CSV file without column names
    writedlm("test/ChebfunTests/YRoots_results/" * file_name * ".csv", roots, ',')
end