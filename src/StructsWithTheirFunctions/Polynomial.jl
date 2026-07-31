using RecursiveArrayTools

function clean_coeff(coeff)
    """Trims trailing all-zero slices (the highest-degree coefficients) along each axis,
    mirroring the Python yroots Polynomial.clean_coeff. Returns the trimmed array. Never
    shrinks an axis below length 1. Because only all-zero slices are removed, trimming one
    axis cannot change another axis's all-zero status, so a single pass per axis suffices."""
    c = coeff
    for ax in 1:ndims(c)
        while size(c, ax) > 1 && all(iszero, selectdim(c, ax, size(c, ax)))
            c = copy(selectdim(c, ax, 1:size(c, ax)-1))
        end
    end
    return c
end

struct MultiPower
    """Contains the coeffs array for a MultiPower object"""
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

struct MultiCheb
    """Contains the coeffs array for a MultiCheb object"""
    coeff
    dim

    function MultiCheb(coeff; clean_zeros=true)
        c = to_julia(coeff)
        if clean_zeros
            c = clean_coeff(c)
        end
        new(c, ndims(c))
    end
end

function to_python(A)
    """" Converts A to an array who's indices match python indices
        So B[i,j,k,...] would return the same thing as P[i,j,k,...],
        where P is the python representation of the matrix A 
    """
    s = size(A)
    dim = length(s)
    B = permutedims(reshape(A,reverse(s)),dim:-1:1)
    return B
end

function to_julia(A)
    """" Converts A to an array who's indices match python indices
        So B[i,j,k,...] would return the same thing as P[i,j,k,...],
        where P is the python representation of the matrix A 
    """

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

function multipower_to_cheb(coeffs)
    """ Takes in a multipower coefficient matrix
        Returns the chebyshev coefficient matrix """
    function get_new_As(As)
        """ Finds the next transformation coefficients (Bs) from the previous ones (As).
            So if x^n = sum(As[i]*T_i(x)), x^(n+1) = sum(Bs[i]*T_i(x)).
        """
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


