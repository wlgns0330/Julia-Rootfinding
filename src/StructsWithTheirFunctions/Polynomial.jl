"""Trims trailing all-zero slices (the highest-degree coefficients) along each axis,
mirroring the Python yroots Polynomial.clean_coeff. Returns the trimmed array. Never
shrinks an axis below length 1. Because only all-zero slices are removed, trimming one
axis cannot change another axis's all-zero status, so a single pass per axis suffices."""
function clean_coeff(coeff)
    c = coeff
    for ax in 1:ndims(c)
        while size(c, ax) > 1 && all(iszero, selectdim(c, ax, size(c, ax)))
            c = copy(selectdim(c, ax, 1:size(c, ax)-1))
        end
    end
    return c
end

"""
    MultiPower(coeff; clean_zeros=true)

A polynomial in the power basis, wrapping its coefficient array so it can be passed
straight to [`solve`](@ref) instead of being approximated from a callable.

`coeff[i, j, ...]` is the coefficient of `x^(i-1) * y^(j-1) * ...`, so `coeff[1, 1]` is
the constant term. The number of axes is the number of variables, available afterwards as
the `dim` field.

`clean_zeros=true` trims trailing all-zero slices, so a padded array does not inflate the
reported degree. Pass `false` to keep the array's shape as given.

# Examples
```julia
julia> p = MultiPower([1.0 0.0; 2.0 3.0]);   # 1 + 2x + 3xy

julia> p.dim
2

julia> eval_MultiPower(p, [0.5, 0.3])
2.45
```
"""
struct MultiPower
    coeff
    dim

    function MultiPower(coeff; clean_zeros=true)
        c = to_julia(coeff)
        if clean_zeros
            c = clean_coeff(c)
        end
        new(c, ndims(c))
    end
end

"""Puts a Chebyshev coefficient array into the layout the solver works in.

The solver reads axis k of a coefficient tensor as variable `ndims - k + 1` -- the reverse of the
`coeff[i,j,...]` <-> `T_i(x) T_j(y) ...` convention a caller writes, and that Python yroots uses.

MultiPower reaches that layout by a longer route: `to_julia` followed by the final `permutedims`
inside `multipower_to_cheb`, which compose to exactly this reversal (checked in 2 through 6
dimensions). MultiCheb has no basis conversion to ride along with, and `to_julia` alone does not
get there, so it reverses directly.

Before this MultiCheb did neither, and was solved transposed -- silently, and only relative to
MultiPower, so a system mixing the two solved one polynomial against the other's transpose."""
to_solver_layout(A) = ndims(A) < 2 ? A : permutedims(A, ndims(A):-1:1)

"""
    MultiCheb(coeff; clean_zeros=true)

A polynomial in the Chebyshev basis, wrapping its coefficient array so it can be passed
straight to [`solve`](@ref) instead of being approximated from a callable.

`coeff[i, j, ...]` is the coefficient of `T_(i-1)(x) * T_(j-1)(y) * ...`, where `T_n` is
the Chebyshev polynomial of the first kind, so `coeff[1, 1]` is the constant term. The
number of axes is the number of variables, available afterwards as the `dim` field.

`clean_zeros=true` trims trailing all-zero slices, so a padded array does not inflate the
reported degree. Pass `false` to keep the array's shape as given.

# Examples
```julia
julia> p = MultiCheb([0.0, 0.0, 1.0]);       # T2(x) = 2x^2 - 1

julia> eval_MultiCheb(p, [0.5])
-0.5

julia> q = MultiCheb([1.0 0.0; 0.0 2.0]);    # 1 + 2*T1(x)*T1(y)

julia> eval_MultiCheb(q, [0.5, 0.3])
1.3
```
"""
struct MultiCheb
    coeff
    dim

    function MultiCheb(coeff; clean_zeros=true)
        c = to_solver_layout(coeff)
        if clean_zeros
            c = clean_coeff(c)
        end
        new(c, ndims(c))
    end
end

"""" Converts A to an array who's indices match python indices
    So B[i,j,k,...] would return the same thing as P[i,j,k,...],
    where P is the python representation of the matrix A 
"""
function to_python(A)
    s = size(A)
    dim = length(s)
    B = permutedims(reshape(A,reverse(s)),dim:-1:1)
    return B
end

"""" Converts A to an array who's indices match python indices
    So B[i,j,k,...] would return the same thing as P[i,j,k,...],
    where P is the python representation of the matrix A 
"""
function to_julia(A)

    s = size(A)
    dim = length(s)
    if dim < 2
        return A # A 1D array is already in the correct order; nothing to permute.
    end
    dim_order = collect(dim:-1:1)
    dim_order[1] = dim-1
    dim_order[2] = dim
    # println(dim_order)
    B = permutedims(A,dim_order)
    return B
end


"""
    eval_MultiPower(multiPower, points)

Evaluate a [`MultiPower`](@ref) at one point or at many.

`points` is either a single point as a length-`dim` vector, or a `dim x npoints` matrix
whose columns are the points. A single point returns a scalar; several return one value
per column.

# Examples
```julia
julia> p = MultiPower([1.0 0.0; 2.0 3.0]);   # 1 + 2x + 3xy

julia> eval_MultiPower(p, [0.5, 0.3])
2.45
```

!!! warning "One-dimensional MultiPower"
    This does not currently work for a `MultiPower` in a single variable: it throws
    `ArgumentError: no valid permutation of dimensions`. `eval_MultiCheb` handles the
    one-variable case correctly.
"""
function eval_MultiPower(multiPower,points)
    function polyval(x, cc)
        cc = collect(eachslice(cc,dims=ndims(cc)))
        c0 = cc[end]
        for i in 1:(length(cc) - 1)
            c0 = c0.*x .+ cc[end-i]
        end
        return c0
    end
    
    if ndims(points) == 1
        if multiPower.dim > 1
            points = reshape(points,(size(points)[end],1))
        else
            points = reshape(points,(1,size(points)[end]))
        end
    end

    if size(points)[end-1] != multiPower.dim
        1/0
    end

    n = multiPower.dim
    c = permutedims(multiPower.coeff,(2,1,collect(3:n)...))
    cc = reshape(c,(ntuple(i->1, ndims(points))..., size(c)...))
    c = polyval(points[1,:],cc)
    for i in 2:n
        c = polyval(points[i,:],c)
    end
    if length(c) == 1
        return c[1]
    else
        return c
    end

end

"""
    eval_MultiCheb(multiCheb, points)

Evaluate a [`MultiCheb`](@ref) at one point or at many.

`points` is either a single point as a length-`dim` vector, or a `dim x npoints` matrix
whose columns are the points. A single point returns a scalar; several return a vector of
values, one per column.

# Examples
```julia
julia> p = MultiCheb([0.0, 0.0, 1.0]);       # T2(x) = 2x^2 - 1

julia> eval_MultiCheb(p, [0.5])
-0.5

julia> eval_MultiCheb(p, [0.5 0.1])
2-element Vector{Float64}:
 -0.5
 -0.98
```
"""
function eval_MultiCheb(multiCheb,points)
    function chebval(x, cc)
        cc = collect(eachslice(cc,dims=ndims(cc)))
        len = length(cc)
        if len == 1
            c0 = cc[1]
            c1 = zero(eltype(c0))
        elseif len == 2
            c0 = cc[1]
            c1 = cc[2]
        else
            x2 = 2 .* x
            c0 = cc[end-1]
            c1 = cc[end]
            for i in 3:len
                tmp = c0
                c0 = cc[end-i+1] .- c1
                c1 = tmp .+ c1 .* x2
            end
        end
        return c0 .+ c1 .* x
    end

    if ndims(points) == 1
        if multiCheb.dim > 1
            points = reshape(points,(size(points)[end],1))
        else
            points = reshape(points,(1,size(points)[end]))
        end
    end

    if size(points)[end-1] != multiCheb.dim
        throw(DimensionMismatch("points has $(size(points)[end-1]) rows but the polynomial is in $(multiCheb.dim) variables"))
    end

    n = multiCheb.dim
    c = reshape(multiCheb.coeff,(ntuple(i->1, ndims(points))..., size(multiCheb.coeff)...))
    for i in 1:n
        c = chebval(points[i,:],c)
    end
    if length(c) == 1
        return c[1]
    else
        return vec(c)
    end

end

""" Takes in a multipower coefficient matrix
    Returns the chebyshev coefficient matrix """
function multipower_to_cheb(coeffs)
    """ Finds the next transformation coefficients (Bs) from the previous ones (As).
        So if x^n = sum(As[i]*T_i(x)), x^(n+1) = sum(Bs[i]*T_i(x)).
    """
    function get_new_As(As)
        n = length(As)
        if n == 0
            return [1.]
        end
        Bs = zeros(n+1)
        # Edge case if As has length 1
        if n == 1
            Bs[2] = As[1]
            return Bs
        end
        # Put in the first and last coeffs
        if n%2 == 0
            Bs[1] = As[2]/2
        end
        Bs[end] = As[end]/2
        # Put in the second coeff
        if n == 2
            Bs[2] = As[1]
            return Bs
        end
        if n%2 == 1
            Bs[2] = As[1] + As[3]/2
        end
        # Do all the middle coefficients, only editing the ones that shouldn't be 0.
        if n > 3
            Bs[3+n%2:2:end-1] = (As[2+n%2:2:end-1] + As[4+n%2:2:end])/2 
        end
        return Bs
    end
    function to_cheb1D(coeffs)
        cheb_coeffs = zeros(eltype(coeffs), size(coeffs))
        As = Float64[]
        n1 = size(coeffs, 1)
        for i in 1:n1
            As = get_new_As(As)
            coeff_i = selectdim(coeffs, 1, i)   # view, no copy
            for k in 1:i
                selectdim(cheb_coeffs, 1, k) .+= As[k] .* coeff_i
            end
        end
        return cheb_coeffs
    end
    function to_chebND(coeffs, dim)
        nd = ndims(coeffs)
        order = Tuple(vcat(dim, [i for i in 1:nd if i != dim]))
        backOrder = Vector{Int}(undef, nd)
        for (i, o) in enumerate(order); backOrder[o] = i; end
        rotated     = PermutedDimsArray(coeffs, order)  # lazy view — no copy
        transformed = to_cheb1D(rotated)
        return permutedims(transformed, backOrder)
    end
    

    cheb_coeffs = coeffs
    for dim in 1:ndims(coeffs)
        # Go through each dimension and transform
        cheb_coeffs = to_chebND(cheb_coeffs,dim)
    end
    # A univariate polynomial has no axes to swap, and permutedims rejects [2,1] on a vector.
    # (The swap is one half of the reversal into the solver's layout; the other half is in
    # `to_julia`. Both are the identity in one dimension.)
    ndims(coeffs) < 2 && return cheb_coeffs
    final_order = append!([2,1],collect(3:ndims(coeffs)))
    return permutedims(cheb_coeffs,final_order)
end


function chebval(x, cc)
    len = length(cc)
    if len == 1
        c0 = cc[1]
        c1 = zeros_like(c0)
    elseif len == 2
        c0 = cc[1]
        c1 = cc[2]
    else
        x2 = 2*x
        c0 = cc[end-1]
        c1 = cc[end]
        for i in 3:len
            tmp = c0
            c0 = cc[end-i+1] - c1
            c1 = tmp + c1*x2
        end
    end
    return c0 + c1*x
end


