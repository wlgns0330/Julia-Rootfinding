include("../src/StructsWithTheirFunctions/Polynomial.jl")
using Test

function test_all_Polynomial()
    @testset "All tests in Polynomial.jl" begin
        test_construction()
    end
end

function test_construction()
    @testset "Constructor tests" begin
        arr1 = [1;3;;5.0;3]

        MC = MultiCheb(arr1)
        @test typeof(MC) == MultiCheb
        # MultiCheb reverses the axes on the way in, to the layout the solver reads. This used to
        # assert the array came back untouched, which is precisely the bug that made a MultiCheb
        # solve as its own transpose.
        @test isapprox(MC.coeff, permutedims(arr1, ndims(arr1):-1:1))

        MP_1 = MultiPower(arr1)
        @test typeof(MP_1) == MultiPower
        @test isapprox(MP_1.coeff,arr1)
        @test 2 == MP_1.dim

        MP_2 = MultiPower([1.4,2])
        @test 1 == MP_2.dim
    end
end

function test_eval_MultiPower()
    @testset "Evaluation tests" begin
        arr1 = [4.0;5;1;3;;3.;2;1;1;;.5;3;2;1]
        MP_1 = MultiPower(arr1)
        points_1 = [4.; 2.5;;5.; 3;;1.0; 1.0;;0.; 0.]
        eval_1 = eval_MultiPower(MP_1, points_1)
        expected_1 = [767.125, 1696.5, 26.5, 4.0]
        for (num, exp) in zip(eval_1, expected_1)
            @test isapprox(num,exp)
        end

        arr2 = [4.0;5;1;3;;3.;2;1;1;;.5;3;2;1;;;0.0;3.4;14;2;;1;1;1;0;;0;24.;3;1]
        MP_2 = MultiPower(arr2)
        points_2 = [4.;-2.5;9;;5.;3;1;;-1.0;1.0;1;;0.;0.;0]
        eval_2 = eval_MultiPower(MP_2, points_2)
        expected_2 = [45260.525, 1494.5, -23.9, 4.0]
        for (num, exp) in zip(eval_2, expected_2)
            @test isapprox(num,exp)
        end

        arr3 = [4.0;3;3.12]
        MP_3 = MultiPower(arr3)
        points_3 = [4.9;16;4;0.0;0;56;-4/5; -18]
        eval_3 = eval_MultiPower(MP_3, points_3)
        expected_3 = [93.6112, 850.72, 65.92, 4., 4., 9956.32, 3.5968, 960.88]
        for (num, exp) in zip(eval_3, expected_3)
            @test isapprox(num,exp)
        end

        arr4 = [3.2124,.0000234,0,0,-3.,6.0,5.4,0,0,-12.9584]
        MP_4 = MultiPower(arr4)
        points_4 = [0,-15,-3.2,6.7,123,.000001,.0000480000005]
        eval_4 = eval_MultiPower(MP_4, points_4)
        expected_4 = [3.2124,  498221229000,  459406.747, -351989715, -8.350207860492976e19,  3.2124,  3.2124]
        for (num,exp) in zip(eval_4,expected_4)
            @test isapprox(num,exp)
        end

        arr5 = [3.2124;.0000234;0;0;-3.;6.0;5.4;0;0;-12.9584;;3.2124;.0000234;0;0;-3.;6.0;5.4;0;0;-12.9584;;;3.2124;.0000234;0;0;-3.;6.0;5.4;0;0;-12.9584;;3.2124;.0000234;0;0;-3.;6.0;5.4;0;0;-12.9584;;;;3.2124;.0000234;0;0;-3.;6.0;5.4;0;0;-12.9584;;3.2124;.0000234;0;0;-3.;6.0;5.4;0;0;-12.9584;;;3.2124;.0000234;0;0;-3.;6.0;5.4;0;0;-12.9584;;3.2124;.0000234;0;0;-3.;6.0;5.4;0;0;-12.9584]
        MP_5 = MultiPower(arr5)
        points_5 = [0;2.1;1.3;-5;;0;0;0;0;;7.9;-4.5;-2.3;-1;;.0000123;-.07673;-.1332453;-.89544412]
        eval_5 = eval_MultiPower(MP_5,points_5)
        expected_5 = [180910404 3.2124 509.053598 4.32890399]
        for (num,exp) in zip(eval_5,expected_5)
            @test isapprox(num,exp)
        end
    end
end