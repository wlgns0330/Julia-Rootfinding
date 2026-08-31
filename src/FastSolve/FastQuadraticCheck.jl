using IterTools

function fast_get_fixed_vars(dim)
    " Returns Julia indicies of fixed vars "
    if dim < 2
        return []
    end
    return reduce(vcat,[collect(IterTools.subsets(1:dim,Val{i}())) for i in dim-1:-1:1])
end

function fast_quadraticCheck2D(test_coeff,tol)
    if ndims(test_coeff) != 2
        return false
    end
    interval = [[-1, -1], [1, 1]]

    c = 1.0*fill(0,6)
    shape = reverse(size(test_coeff))
    c[1] = test_coeff[1,1]
    if shape[1] > 1
        c[2] = test_coeff[1,2]
    end
    if shape[2] > 1
        c[3] = test_coeff[2,1]
    end
    if shape[1] > 2
        c[4] = test_coeff[1,3]
    end
    if shape[1] > 1 && shape[2] > 1
        c[5] = test_coeff[2,2]
    end
    if shape[2] > 2
        c[6] = test_coeff[3,1]
    end

    other_sum = sum(abs, test_coeff) - sum(abs.(c)) + tol
    k0 = c[1]-c[4]-c[6]
    k3 = 2*c[4]
    k5 = 2*c[6]
    function fast_eval_func(x,y)
        return k0 + (c[2] + k3 * x + c[5] * y) * x  + (c[3] + k5 * y) * y
    end

    det = 16 * c[4] * c[6] - c[5]^2
    if det != 0
        int_x = (c[3] * c[5] - 4 * c[2] * c[6]) / det
        int_y = (c[2] * c[5] - 4 * c[3] * c[4]) / det
    else                      # det is zero,
        int_x = Inf
        int_y = Inf
    end

    min_satisfied, max_satisfied = false,false
    #Check all the corners
    eval = fast_eval_func(interval[1][1], interval[1][2])
    min_satisfied = min_satisfied || eval < other_sum
    max_satisfied = max_satisfied || eval > -other_sum
    if min_satisfied && max_satisfied
        return false
    end

    eval = fast_eval_func(interval[2][1], interval[1][2])
    min_satisfied = min_satisfied || eval < other_sum
    max_satisfied = max_satisfied || eval > -other_sum
    if min_satisfied && max_satisfied
        return false
    end

    eval = fast_eval_func(interval[1][1], interval[2][2])
    min_satisfied = min_satisfied || eval < other_sum
    max_satisfied = max_satisfied || eval > -other_sum
    if min_satisfied && max_satisfied
        return false
    end

    eval = fast_eval_func(interval[2][1], interval[2][2])
    min_satisfied = min_satisfied || eval < other_sum
    max_satisfied = max_satisfied || eval > -other_sum
    if min_satisfied && max_satisfied
        return false
    end

    #Check the x constant boundaries
    #The partial with respect to y is zero
    #Dy:  c4x + 4c5y = -c2 =>   y = (-c2-c4x)/(4c5)
    if c[6] != 0
        cc5 = 4 * c[6]
        x = interval[1][1]
        y = -(c[3] + c[5]*x)/cc5
        if interval[1][2] < y < interval[2][2]
            eval = fast_eval_func(x,y)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
        x = interval[2][1]
        y = -(c[3] + c[5]*x)/cc5
        if interval[1][2] < y < interval[2][2]
            eval = fast_eval_func(x,y)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
    end

    #Check the y constant boundaries
    #The partial with respect to x is zero
    #Dx: 4c3x +  c4y = -c1  =>  x = (-c1-c4y)/(4c3)
    if c[4] != 0
        cc3 = 4*c[4]
        y = interval[1][2]
        x = -(c[2] + c[5]*y)/cc3
        if interval[1][1] < x < interval[2][1]
            eval = fast_eval_func(x,y)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end

        y = interval[2][2]
        x = -(c[2] + c[5]*y)/cc3
        if interval[1][1] < x < interval[2][1]
            eval = fast_eval_func(x,y)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
    end

    #Check the interior value
    if interval[1][1] < int_x < interval[2][1] && interval[1][2] < int_y < interval[2][2]
        eval = fast_eval_func(int_x,int_y)
        min_satisfied = min_satisfied || eval < other_sum
        max_satisfied = max_satisfied || eval > -other_sum
        if min_satisfied && max_satisfied
            return false
        end
    end

    return true
end

function fast_quadraticCheck3D(test_coeff,tol)
    """One of subinterval_checks

    Finds the min of the absolute value of the quadratic part, and compares to the sum of the
    rest of the terms.  There can't be a root if min(extreme_values) > other_sum	or if
    max(extreme_values) < -other_sum. We can short circuit and finish
    faster as soon as we find one value that is < other_sum and one value that > -other_sum.

    Parameters
    ----------
    test_coeff : array
        The coefficient matrix of the polynomial to check
    tol: float
        The bound of the sup norm error of the chebyshev approximation.

    Returns
    -------
    mask : list
        A list of the results of each interval. false if the function is guarenteed to never be zero
        in the unit box, true otherwise
    """
    if ndims(test_coeff) != 3
        return false
    end

    #Padding is slow, so check the shape instead.
    c = 1.0*fill(0,10)
    shape = reverse(size(test_coeff))
    c[1] = test_coeff[1,1,1]
    if shape[1] > 1
        c[2] = test_coeff[1,1,2]
    end
    if shape[2] > 1
        c[3] = test_coeff[1,2,1]
    end
    if shape[3] > 1
        c[4] = test_coeff[2,1,1]
    end
    if shape[1] > 1 && shape[2] > 1
        c[5] = test_coeff[1,2,2]
    end
    if shape[1] > 1 && shape[3] > 1
        c[6] = test_coeff[2,1,2]
    end
    if shape[2] > 1 && shape[3] > 1
        c[7] = test_coeff[2,2,1]
    end
    if shape[1] > 2
        c[8] = test_coeff[1,1,3]
    end
    if shape[2] > 2
        c[9] = test_coeff[1,3,1]
    end
    if shape[3] > 2
        c[10] = test_coeff[3,1,1]
    end

    #The sum of the absolute values of everything else
    other_sum = sum(abs, test_coeff) - sum(abs.(c)) + tol

    #function for evaluating c0 + c1x + c2y +c3z + c4xy + c5xz + c6yz + c7T_2(x) + c8T_2(y) + c9T_2(z)
    # Use the Horner form because it is much faster, also do any repeated computatons in advance
    k0 = c[1]-c[8]-c[9]-c[10]
    k7 = 2*c[8]
    k8 = 2*c[9]
    k9 = 2*c[10]
    function fast_eval_func(x,y,z)
        return k0 + (c[2] + k7 * x + c[5] * y + c[6] * z) * x + (c[3] + k8 * y + c[7] * z) * y + (c[4] + k9 * z) * z
    end

    #The interior min
    #Comes from solving dx, dy, dz = 0
    #Dx: 4c7x +  c4y +  c5z = -c1    Matrix inverse is  [(16c8c9-c6^2) -(4c4c9-c5c6)  (c4c6-4c5c8)]
    #Dy:  c4x + 4c8y +  c6z = -c2                       [-(4c4c9-c5c6) (16c7c9-c5^2) -(4c6c7-c4c5)]
    #Dz:  c5x +  c6y + 4c9z = -c3                       [(c4c6-4c5c8)  -(4c6c7-c4c5) (16c7c8-c4^2)]
    #These computations are the same for all subintevals, so do them first
    kk7 = 2*k7 #4c7
    kk8 = 2*k8 #4c8
    kk9 = 2*k9 #4c9
    fix_x_det = kk8*kk9-c[7]^2
    fix_y_det = kk7*kk9-c[6]^2
    fix_z_det = kk7*kk8-c[5]^2
    minor_1_2 = kk9*c[5]-c[6]*c[7]
    minor_1_3 = c[5]*c[7]-kk8*c[6]
    minor_2_3 = kk7*c[7]-c[5]*c[6]
    det = 4*c[8]*fix_x_det - c[5]*minor_1_2 + c[6]*minor_1_3
    if det != 0
        int_x = (c[2]*-fix_x_det + c[3]*minor_1_2  + c[4]*-minor_1_3)/det
        int_y = (c[2]*minor_1_2  + c[3]*-fix_y_det + c[4]*minor_2_3)/det
        int_z = (c[2]*-minor_1_3  + c[3]*minor_2_3  + c[4]*-fix_z_det)/det
    else
        int_x = Inf
        int_y = Inf
        int_z = Inf
    end

    interval = [[-1, -1, -1], [1, 1, 1]]
    #easier names for each value...
    x0 = interval[1][1]
    x1 = interval[2][1]
    y0 = interval[1][2]
    y1 = interval[2][2]
    z0 = interval[1][3]
    z1 = interval[2][3]

    min_satisfied, max_satisfied = false,false
    #Check all the corners
    eval = fast_eval_func(x0, y0, z0)
    min_satisfied = min_satisfied || eval < other_sum
    max_satisfied = max_satisfied || eval > -other_sum
    if min_satisfied && max_satisfied
        return false
    end
    eval = fast_eval_func(x1, y0, z0)
    min_satisfied = min_satisfied || eval < other_sum
    max_satisfied = max_satisfied || eval > -other_sum
    if min_satisfied && max_satisfied
        return false
    end
    eval = fast_eval_func(x0, y1, z0)
    min_satisfied = min_satisfied || eval < other_sum
    max_satisfied = max_satisfied || eval > -other_sum
    if min_satisfied && max_satisfied
        return false
    end
    eval = fast_eval_func(x0, y0, z1)
    min_satisfied = min_satisfied || eval < other_sum
    max_satisfied = max_satisfied || eval > -other_sum
    if min_satisfied && max_satisfied
        return false
    end
    eval = fast_eval_func(x1, y1, z0)
    min_satisfied = min_satisfied || eval < other_sum
    max_satisfied = max_satisfied || eval > -other_sum
    if min_satisfied && max_satisfied
        return false
    end
    eval = fast_eval_func(x1, y0, z1)
    min_satisfied = min_satisfied || eval < other_sum
    max_satisfied = max_satisfied || eval > -other_sum
    if min_satisfied && max_satisfied
        return false
    end
    eval = fast_eval_func(x0, y1, z1)
    min_satisfied = min_satisfied || eval < other_sum
    max_satisfied = max_satisfied || eval > -other_sum
    if min_satisfied && max_satisfied
        return false
    end
    eval = fast_eval_func(x1, y1, z1)
    min_satisfied = min_satisfied || eval < other_sum
    max_satisfied = max_satisfied || eval > -other_sum
    if min_satisfied && max_satisfied
        return false
    end
    #Adds the x and y constant boundaries
    #The partial with respect to z is zero
    #Dz:  c5x +  c6y + 4c9z = -c3   => z=(-c3-c5x-c6y)/(4c9)
    if c[10] != 0
        c5x0_c3 = c[6]*x0 + c[4]
        c6y0 = c[7]*y0
        z = -(c5x0_c3+c6y0)/kk9
        if z0 < z < z1
            eval = fast_eval_func(x0,y0,z)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
        c6y1 = c[7]*y1
        z = -(c5x0_c3+c6y1)/kk9
        if z0 < z < z1
            eval = fast_eval_func(x0,y1,z)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
        c5x1_c3 = c[6]*x1 + c[4]
        z = -(c5x1_c3+c6y0)/kk9
        if z0 < z < z1
            eval = fast_eval_func(x1,y0,z)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
        z = -(c5x1_c3+c6y1)/kk9
        if z0 < z < z1
            eval = fast_eval_func(x1,y1,z)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
    end

    #Adds the x and z constant boundaries
    #The partial with respect to y is zero
    #Dy:  c4x + 4c8y + c6z = -c2   => y=(-c2-c4x-c6z)/(4c8)
    if c[9] != 0
        c6z0 = c[7]*z0
        c2_c4x0 = c[3]+c[5]*x0
        y = -(c2_c4x0+c6z0)/kk8
        if y0 < y < y1
            eval = fast_eval_func(x0,y,z0)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
        c6z1 = c[7]*z1
        y = -(c2_c4x0+c6z1)/kk8
        if y0 < y < y1
            eval = fast_eval_func(x0,y,z1)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
        c2_c4x1 = c[3]+c[5]*x1
        y = -(c2_c4x1+c6z0)/kk8
        if y0 < y < y1
            eval = fast_eval_func(x1,y,z0)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
        y = -(c2_c4x1+c6z1)/kk8
        if y0 < y < y1
            eval = fast_eval_func(x1,y,z1)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
    end

    #Adds the y and z constant boundaries
    #The partial with respect to x is zero
    #Dx: 4c7x +  c4y +  c5z = -c1   => x=(-c1-c4y-c5z)/(4c7)
    if c[8] != 0
        c1_c4y0 = c[2]+c[5]*y0
        c5z0 = c[6]*z0
        x = -(c1_c4y0+c5z0)/kk7
        if x0 < x < x1
            eval = fast_eval_func(x,y0,z0)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
        c5z1 = c[6]*z1
        x = -(c1_c4y0+c5z1)/kk7
        if x0 < x < x1
            eval = fast_eval_func(x,y0,z1)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
        c1_c4y1 = c[2]+c[5]*y1
        x = -(c1_c4y1+c5z0)/kk7
        if x0 < x < x1
            eval = fast_eval_func(x,y1,z0)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
        x = -(c1_c4y1+c5z1)/kk7
        if x0 < x < x1
            eval = fast_eval_func(x,y1,z1)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
    end

    #Add the x constant boundaries
    #The partials with respect to y and z are zero
    #Dy:  4c8y +  c6z = -c2 - c4x    Matrix inverse is [4c9  -c6]
    #Dz:   c6y + 4c9z = -c3 - c5x                      [-c6  4c8]
    if fix_x_det != 0
        c2_c4x0 = c[3]+c[5]*x0
        c3_c5x0 = c[4]+c[6]*x0
        y = (-kk9*c2_c4x0 +   c[7]*c3_c5x0)/fix_x_det
        z = (c[7]*c2_c4x0 -    kk8*c3_c5x0)/fix_x_det
        if y0 < y < y1 && z0 < z < z1
            eval = fast_eval_func(x0,y,z)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
        c2_c4x1 = c[3]+c[5]*x1
        c3_c5x1 = c[4]+c[6]*x1
        y = (-kk9*c2_c4x1 +   c[7]*c3_c5x1)/fix_x_det
        z = (c[7]*c2_c4x1 -    kk8*c3_c5x1)/fix_x_det
        if y0 < y < y1 && z0 < z < z1
            eval = fast_eval_func(x1,y,z)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
    end

    #Add the y constant boundaries
    #The partials with respect to x and z are zero
    #Dx: 4c7x +  c5z = -c1 - c40    Matrix inverse is [4c9  -c5]
    #Dz:  c5x + 4c9z = -c3 - c6y                      [-c5  4c7]
    if fix_y_det != 0
        c1_c4y0 = c[2]+c[5]*y0
        c3_c6y0 = c[4]+c[7]*y0
        x = (-kk9*c1_c4y0 +   c[6]*c3_c6y0)/fix_y_det
        z = (c[6]*c1_c4y0 -    kk7*c3_c6y0)/fix_y_det
        if x0 < x < x1 && z0 < z < z1
            eval = fast_eval_func(x,y0,z)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
        c1_c4y1 = c[2]+c[5]*y1
        c3_c6y1 = c[4]+c[7]*y1
        x = (-kk9*c1_c4y1 +   c[6]*c3_c6y1)/fix_y_det
        z = (c[6]*c1_c4y1 -    kk7*c3_c6y1)/fix_y_det
        if x0 < x < x1 && z0 < z < z1
            eval = fast_eval_func(x,y1,z)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
    end

    #Add the z constant boundaries
    #The partials with respect to x and y are zero
    #Dx: 4c7x +  c4y  = -c1 - c5z    Matrix inverse is [4c8  -c4]
    #Dy:  c4x + 4c8y  = -c2 - c6z                      [-c4  4c7]
    if fix_z_det != 0
        c1_c5z0 = c[2]+c[6]*z0
        c2_c6z0 = c[3]+c[7]*z0
        x = (-kk8*c1_c5z0 +   c[5]*c2_c6z0)/fix_z_det
        y = (c[5]*c1_c5z0 -    kk7*c2_c6z0)/fix_z_det
        if x0 < x < x1 && y0 < y < y1
            eval = fast_eval_func(x,y,z0)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
        c1_c5z1 = c[2]+c[6]*z1
        c2_c6z1 = c[3]+c[7]*z1
        x = (-kk8*c1_c5z1 +   c[5]*c2_c6z1)/fix_z_det
        y = (c[5]*c1_c5z1 -    kk7*c2_c6z1)/fix_z_det
        if x0 < x < x1 && y0 < y < y1
            eval = fast_eval_func(x,y,z1)
            min_satisfied = min_satisfied || eval < other_sum
            max_satisfied = max_satisfied || eval > -other_sum
            if min_satisfied && max_satisfied
                return false
            end
        end
    end

    #Add the interior value
    if x0 < int_x < x1 && y0 < int_y < y1 && z0 < int_z < z1
        eval = fast_eval_func(int_x,int_y,int_z)
        min_satisfied = min_satisfied || eval < other_sum
        max_satisfied = max_satisfied || eval > -other_sum
        if min_satisfied && max_satisfied
            return false
        end
    end

    # No root possible
    return true

end

function fast_quadraticCheckND(test_coeff, tol)
    """One of subinterval_checks

    Finds the min of the absolute value of the quadratic part, and compares to the sum of the
    rest of the terms. There can't be a root if min(extreme_values) > other_sum	or if
    max(extreme_values) < -other_sum. We can short circuit and finish
    faster as soon as we find one value that is < other_sum and one value that > -other_sum.

    Parameters
    ----------
    test_coeff_in : numpy array
        The coefficient matrix of the polynomial to check
    tol: float
        The bound of the sup norm error of the chebyshev approximation.

    Returns
    -------
    True if there is guaranteed to be no root in the interval, False otherwise
    """
    #get the dimension and make sure the coeff tensor has all the right
    # quadratic coeff spots, set to zero if necessary
    dim = ndims(test_coeff)
    init_dims = size(test_coeff)
    dims = max.(init_dims,3)
    new_coeff = zeros(dims)
    new_coeff[[1:init_dims[i] for i in 1:dim]...] = test_coeff
    test_coeff = new_coeff
    interval = hcat(fill(-1,dim),fill(1,dim))

    quad_coeff = zeros(tuple(3*Int.(ones(dim))...))
    #A and B are arrays for slicing
    A = zeros((dim,dim))
    B = zeros(dim)
    pure_quad_coeff = zeros(dim)
    const1=0
    for spot in IterTools.product([1:3 for i in 1:dim]...)
        spot_deg = sum(spot)-dim
        if spot_deg == 1
            #coeff of linear terms
            i = [idx for idx in 1:dim if reverse(spot)[idx]!= 1][1]
            B[i] = copy(test_coeff[spot...])
            quad_coeff[spot...] = test_coeff[spot...]
            test_coeff[spot...] = 0
        elseif spot_deg == 0
            #constant term
            const1 = copy(test_coeff[spot...])
            quad_coeff[spot...] = const1
            test_coeff[spot...] = 0
        elseif spot_deg < 3
            where_nonzero = [idx for idx in 1:dim if reverse(spot)[idx]!= 1]
            if length(where_nonzero) == 2
                #coeff of cross terms
                i,j = where_nonzero
                #with symmetric matrices, we only need to store the lower part
                A[i,j] = copy(test_coeff[spot...])
                A[j,i] = A[i,j]
                #todo: see if we can store this in only one half of A

            else
                #coeff of pure quadratic terms
                i = where_nonzero[1]
                pure_quad_coeff[i] = copy(test_coeff[spot...])
            end
        quad_coeff[spot...] = test_coeff[spot...]
        test_coeff[spot...] = 0
        end
    end
    pure_quad_coeff_doubled = [p*2 for p in pure_quad_coeff]
    A[diagind(A)] = [p*2 for p in pure_quad_coeff_doubled]

    #create a poly object for evals
    k0 = const1 - sum(pure_quad_coeff)

    function fast_eval_func(point)
        "fast evaluation of quadratic chebyshev polynomials using horner's algorithm"
        _sum = k0
        for i in 1:dim
            coord = point[i]
            _sum += (B[i] + pure_quad_coeff_doubled[i]*coord + (i<dim ? sum([A[j,i]*point[j] for j=i+1:dim]) : 0)) * coord
        end
        return _sum
    end

    #The sum of the absolute values of everything else
    other_sum = sum(abs, test_coeff) .+ tol

    #iterator for sides
    fixed_vars = fast_get_fixed_vars(dim)


    Done = false
    min_satisfied, max_satisfied = false,false
    #fix all variables--> corners
    for corner in IterTools.product([1:2 for i in 1:dim]...)
        #j picks if upper/lower bound. i is which var
        eval = fast_eval_func([interval[i,corner[i]] for i in 1:dim])
        min_satisfied = min_satisfied || eval < other_sum
        max_satisfied = max_satisfied || eval > -other_sum
    
        if min_satisfied && max_satisfied
            Done = true
            break
        end
    end
    if !Done
        X = zeros(dim)
        for fixed in fixed_vars
            #fixed some variables --> "sides"
            #we only care about the equations from the unfixed variables
            unfixed = deleteat!(collect(1:dim), fixed)
            fixed_args = [item for item in fixed]
            A_ = A[:,unfixed][unfixed,:]

            #if diagonal entries change sign, can't be definite
            diag_vals = diag(A_)
            len = length(diag_vals)
            sign_change = false
            for i in 1:len-1
                c = diag_vals[i]
                #sign change?
                if c*diag_vals[i+1]<0
                    sign_change = true
                    break
                end
            end
            #if no sign change, can find extrema
            if !sign_change
                #not full rank --> no soln
                if rank(A_) == size(A_)[1]
                    fixed_A = A[:,unfixed][fixed_args,:]
                    B_ = B[unfixed]
                    for side in IterTools.product([1:2 for i in 1:length(fixed)]...)
                        X0 = [interval[i,side[i]] for i in 1:length(side)]
                        X_ = A_\(-B_-(fixed_A')*X0)
                        #make sure it's in the domain
                        do_next = true
                        for i in 1:length(unfixed)
                            var = unfixed[i]
                            if interval[var,1] <= X_[i] <= interval[var,2]
                                continue
                            else
                                do_next = false
                                break
                            end
                        end
                        if do_next
                            X[fixed_args] = X0
                            X[unfixed] = X_
                            eval = fast_eval_func(X)
                            min_satisfied = min_satisfied || eval < other_sum
                            max_satisfied = max_satisfied || eval > -other_sum
                            if min_satisfied && max_satisfied
                                Done = true
                                break
                            end
                        end
                    end
                end
            end
            if Done
                break
            end
        end
        if !Done
            #fix no vars--> interior
            #if diagonal entries change sign, can't be definite
            should_next = true
            for i in 1:length(pure_quad_coeff)-1
                c = pure_quad_coeff[i]
                #sign change?
                if c*pure_quad_coeff[i+1]<0
                    should_next = false
                    break
                end
            end
            #if no sign change, can find extrema
            if should_next
                #not full rank --> no soln
                if rank(A) == size(A)[1]
                    X = A\-B
                    #make sure it's in the domain
                    do_next = true
                    for i in 1:dim
                        if interval[i,1] <= X[i] <= interval[i,2]
                            continue
                        else
                            do_next = false
                            break
                        end
                    end
                    if do_next
                        curr_eval = fast_eval_func(X)
                        min_satisfied = min_satisfied || curr_eval < other_sum
                        max_satisfied = max_satisfied || curr_eval > -other_sum
                        if min_satisfied && max_satisfied
                            Done = true
                        end
                    end
                end
            end
        end
    end
    return !Done
end

function fast_quadraticCheck(test_coeff,tol,nd_check=false)
    if ndims(test_coeff) == 2 && !nd_check
        return fast_quadraticCheck2D(test_coeff, tol)
    elseif ndims(test_coeff) == 3 && !nd_check
        return fast_quadraticCheck3D(test_coeff, tol)
    else
        return fast_quadraticCheckND(test_coeff, tol)
    end
end
