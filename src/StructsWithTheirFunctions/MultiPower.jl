

""" Gets the n-d slices needed to slice a matrix into the top corner of another.

Parameters
----------
matrix_shape : tuple.
    The matrix shape of interest.
Returns
-------
slices : list
    Each value of the list is a slice of the matrix in some dimension. It is exactly the size of matrix_shape.
"""
function slice_top(matrix_shape)
    slices = []
    for i in matrix_shape
        push!(slices,1:i)
    end
    return slices
end

"""
Matches the shape of two matrixes.

Parameters
----------
a, b : ndarray
    Matrixes whose size is to be matched.

Returns
-------
a, b : ndarray
    Matrixes of equal size.
"""
function match_size(a,b)
    a_shape = size(a)
    b_shape = size(b)
    dim = length(a_shape)
    new_shape = zeros(Int64, length(a_shape))
    for i in 1:dim
        new_shape[i] = Int64(max(a_shape[i], b_shape[i]))
    end
    a_new = zeros(new_shape...)
    a_new[slice_top(a_shape)...] = a
    b_new = zeros(new_shape...)
    b_new[slice_top(b_shape)...] = b
    return a_new, b_new
end

function polyval(x, cc)
    dim = length(size(cc))
    slices = collect(eachslice(cc,dims=dim))
    println(slices)
    c0 = slices[end]
    println(c0)
    for i in 1:length(slices)-1
        println(slices[end-i])
        println(c0.*x)
        c0 = slices[end-i] + c0.*x
    end
    return c0
end
