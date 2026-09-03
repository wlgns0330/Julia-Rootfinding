# TrackedInterval.jl does not stand alone: getFinalInterval calls twoProd, which lives in
# ChebyshevSubdivisionSolver.jl. Including the solver pulls in TrackedInterval.jl itself
# (and SolverOptions.jl and QuadraticCheck.jl) transitively, so this covers both.
include("../../Julia-Rootfinding/src/ChebyshevSubdivisionSolver.jl")
using Test

# TrackedInterval.jl reads two globals that solve() normally sets on the way in
# (CombinedSolver.jl sets `type` and `precision` from its `roundoff` argument). Nothing
# in this file goes through solve(), so they are set here. Without them the constructor
# throws UndefVarError on `type` at TrackedInterval.jl:53 and every test in this file
# errors -- which is what happened whenever this suite ran before
# ChebyshevSubdivisionSolverTest.jl, the only other file that sets them.
function setupTrackedIntervalGlobals()
    global type = Float64
    global precision = 53
end

function test_all_TrackedInterval()
    @testset "All tests in TrackedIntervalTest.jl" begin
        setupTrackedIntervalGlobals()
        test_copyInterval()
        test_addTransform()
        test_getIntervalForCombining()
        test_isPoint()
        test_getFinalInterval()
        test_getFinalPoint()
        test_contains()
        test_overlapsWith()
        test_startFinalStep()
    end
end

function test_copyInterval()
    @testset "copyInterval unit tests" begin
        trackedInterval_1 = TrackedInterval([1;2;;-1;1;;-3;2])
        copiedInterval_1 = copyInterval(trackedInterval_1)
        # BASIC INTERVAL COPYING
        @test copiedInterval_1.interval == trackedInterval_1.interval
        @test copiedInterval_1.empty == false
        @test copiedInterval_1.nextTransformPoints == fill(0.0394555475981047,3)
        trackedInterval_1.empty = true
        trackedInterval_1.transforms = [[2;3;;4;5;;1.234234;1.3212331]]
        trackedInterval_1.nextTransformPoints[2] = 4
        copiedInterval_2 = copyInterval(trackedInterval_1)
        # TEST OTHER ATTRIBUTES COPIED
        @test copiedInterval_2.empty == true
        @test copiedInterval_2.transforms == [[2;3;;4;5;;1.234234;1.3212331]]
        @test copiedInterval_2.nextTransformPoints[1] == 0.0394555475981047
        @test copiedInterval_2.nextTransformPoints[2] == 4

        # FINAL STEP TEST
        @test copiedInterval_2.finalStep == false
        @test copiedInterval_2.possibleDuplicateRoots == []
        @test copiedInterval_2.possibleExtraRoot == false
        @test copiedInterval_2.preFinalInterval == []
        @test copiedInterval_2.preFinalTransforms == []
        trackedInterval_1.finalStep = true
        trackedInterval_1.possibleDuplicateRoots = [3,1,2]
        trackedInterval_1.possibleExtraRoot = true
        trackedInterval_1.preFinalInterval = [1;2;;-1;1;;-3;2]
        trackedInterval_1.preFinalTransforms = [[1;2;;-1;1;;-3;2]]
        copiedInterval_3 = copyInterval(trackedInterval_1)
        @test copiedInterval_3.finalStep == true
        @test copiedInterval_3.possibleDuplicateRoots == [3,1,2]
        @test copiedInterval_3.possibleExtraRoot == true
        @test copiedInterval_3.preFinalInterval == [1;2;;-1;1;;-3;2]
        @test copiedInterval_3.preFinalTransforms == [[1;2;;-1;1;;-3;2]]


    end
end

function test_addTransform()
    @testset "addTransform unit tests" begin
        # Empty interval test 1
        trackedInterval_1 = TrackedInterval([-1.;2;;-1;1])
        trackedInterval_1.finalStep = true
        trackedInterval_1.canThrowOutFinalStep = true
        subInterval_1 = [0;-.0001;;-1;1]
        addTransform(trackedInterval_1,subInterval_1)
        @test trackedInterval_1.empty == true
        @test trackedInterval_1.transforms == []
        @test trackedInterval_1.interval == [-1.;2;;-1;1]

        # Empty interval test 2
        trackedInterval_2 = TrackedInterval([-1;2;;-1;1])
        subInterval_2 = [0;-.0001;;-1;1]
        addTransform(trackedInterval_2,subInterval_2)
        @test trackedInterval_2.empty == true
        @test trackedInterval_2.transforms == []
        @test trackedInterval_2.interval == [-1.;2;;-1;1]

        trackedInterval_3 = TrackedInterval([-1.;2;;-1;1])
        trackedInterval_3.finalStep = true
        subInterval_3 = [0;-.0001;;-1;1]
        addTransform(trackedInterval_3,subInterval_3)
        @test trackedInterval_3.empty == false
        @test isapprox(trackedInterval_3.interval,[ 0.5;  0.5;;-1.;   1. ])
    
        trackedInterval_4 = TrackedInterval([-1.;2;;-1;1])
        trackedInterval_4.finalStep = true
        subInterval_4 = [.0001;0;;-1;1]
        addTransform(trackedInterval_4,subInterval_4)
        @test trackedInterval_4.empty == false
        @test isapprox(trackedInterval_4.interval,[ 0.50015;  0.50015;;-1.;       1.     ])
        @test isapprox(trackedInterval_4.transforms[1],[0.e+00; 1.e+00;;1.e-04; 0.e+00])

        trackedInterval_5 = TrackedInterval([-1.;2;;-1;1;;-1;.9])
        trackedInterval_5.finalStep = true
        subInterval_5 = [.0001;0;;-1;1;;-1;1]
        addTransform(trackedInterval_5,subInterval_5)
        @test trackedInterval_5.empty == false
        @test isapprox(trackedInterval_5.interval,[ 0.50015;  0.50015;;-1.;       1.     ;;-1.;       0.9    ])
        @test isapprox(trackedInterval_5.transforms[1],[0.e+00; 1.e+00; 1.e+00;;1.e-04; 0.e+00; 0.e+00])

        trackedInterval_6 = TrackedInterval([-1.;2;;-1;1;;-1;.9])
        trackedInterval_6.finalStep = true
        subInterval_6 = [-18;-18.01;;-3;1;;-3;1]
        addTransform(trackedInterval_6,subInterval_6)
        @test trackedInterval_6.empty == false
        @test isapprox(trackedInterval_6.interval,[-1.;  -1. ;;-1.;   1. ;;-1.;   0.9])
        @test isapprox(trackedInterval_6.transforms[1],[ 0.;  1.;  1.;;-1.;  0.;  0.])

        trackedInterval_7 = TrackedInterval([-1.;2;;-1;1;;-1;.9])
        trackedInterval_7.finalStep = true
        subInterval_7 = [3;-3.01;;3;1;;3;1]
        addTransform(trackedInterval_7,subInterval_7)
        @test trackedInterval_7.empty == false
        @test isapprox(trackedInterval_7.interval,[2.;  2. ;;1.;  1. ;;0.9; 0.9])
        @test isapprox(trackedInterval_7.transforms[1],[0.; 0.; 0.;;1.; 1.; 1.])

    end
end

function test_getIntervalForCombining()
    @testset "getIntervalForCombining unit tests" begin
        trackedInterval = TrackedInterval([-1;1;;-1.2332;1.2134;;-5;1])
        @test isapprox(getIntervalForCombining(trackedInterval),[-1;1;;-1.2332;1.2134;;-5;1])
        @test trackedInterval.preFinalInterval == []
        trackedInterval.preFinalInterval = [-5;5;;-1;1;;500;1]
        trackedInterval.finalStep = true
        @test isapprox(getIntervalForCombining(trackedInterval),[-5;5;;-1;1;;500;1])
        trackedInterval.finalStep = false
        @test isapprox(getIntervalForCombining(trackedInterval),[-1;1;;-1.2332;1.2134;;-5;1])
    end
end

function test_isPoint()
    @testset "test_isPoint unit tests" begin
        trackedInterval_1 = TrackedInterval([1;2;;3;4])
        @test !isPoint(trackedInterval_1)

        trackedInterval_2 = TrackedInterval([1;1;;3;4])
        @test !isPoint(trackedInterval_2)

        trackedInterval_3 = TrackedInterval([1;2;;3;3])
        @test !isPoint(trackedInterval_3)

        trackedInterval_4 = TrackedInterval([1;2;;1;2])
        @test !isPoint(trackedInterval_4)

        trackedInterval_5 = TrackedInterval([1;1;;3;3])
        @test isPoint(trackedInterval_5)

        trackedInterval_6 = TrackedInterval([2;2;;3;4;;5;5;;7;7])
        @test !isPoint(trackedInterval_6)

        trackedInterval_7 = TrackedInterval([1;1;;pi;pi;;5;5;;7;7])
        @test isPoint(trackedInterval_7)

        trackedInterval_8 = TrackedInterval([1+10^-20;1;;3-10^-20;3])
        @test isPoint(trackedInterval_8)
    end
end

function test_getFinalInterval()
    @testset "getFinalInterval unit tests" begin
        tInterval_1 = TrackedInterval([-1.;1;;-1;1;;-1;1;;-1;1])
        tInterval_1.interval = [-.5;.11;;-9;5;;-3.;1;;-1.;1]
        push!(tInterval_1.transforms,[.01;.1;.624;1;;.009;.0123847;2.;0])
        tInterval_1.finalStep = false
        tInterval_1.preFinalTransforms = []
        push!(tInterval_1.preFinalTransforms,[.1;.1;.624;1;;.009;.0123847;2.;0])
        @test isapprox(getFinalInterval(tInterval_1),[-1.000000e-03;  1.900000e-02;;-8.761530e-02;  1.123847e-01;; 1.376000e+00;  2.624000e+00;;-1.000000e+00;  1.000000e+00])
        @test isapprox(tInterval_1.finalAlpha, [0.01,  0.1,   0.624, 1.   ])
        @test isapprox(tInterval_1.finalBeta, [0.009,     0.0123847, 2.,        0.       ])
        tInterval_1.finalStep = true
        @test isapprox(getFinalInterval(tInterval_1),[-0.091;      0.109;;-0.0876153;  0.1123847;;1.376;      2.624;;-1.;         1.       ])
        @test isapprox(tInterval_1.finalAlpha,[0.1,   0.1,   0.624, 1.   ])
        @test isapprox(tInterval_1.finalBeta,[0.009,     0.0123847, 2.,        0.       ])

        tInterval_2 = TrackedInterval([-2.;1;;])
        tInterval_2.interval = [-.9-.89;;]
        push!(tInterval_2.transforms,[.4;;4])
        tInterval_2.finalStep = true
        tInterval_2.preFinalTransforms = []
        push!(tInterval_2.preFinalTransforms,[1.;;0.])
        @test isapprox(getFinalInterval(tInterval_2),[-2.;  1.;;])
        @test isapprox(tInterval_2.finalAlpha, [1.5])
        @test isapprox(tInterval_2.finalBeta, [-.5])
        tInterval_2.finalStep = false
        @test isapprox(getFinalInterval(tInterval_2),[3.2; 4.4;;])
        @test isapprox(tInterval_2.finalAlpha, [.6])
        @test isapprox(tInterval_2.finalBeta, [3.8])
    end
end

function test_getFinalPoint()
    @testset "getFinalPoint unit tests" begin
        tInterval_1 = TrackedInterval([-1.;1;;-1;1;;-1;1;;-1;1])
        tInterval_1.interval = [-.5;.11;;-9;5;;-3.;1;;-1.;1]
        tInterval_1.finalInterval = [-1.;1;;-1;1;;-1;1;;-1;1]
        push!(tInterval_1.transforms,[.01;.1;.624;1;;.009;.0123847;2.;0])
        tInterval_1.finalStep = false
        tInterval_1.preFinalTransforms = []
        push!(tInterval_1.preFinalTransforms,[.1;.1;.624;1;;.009;.0123847;2.;9])
        @test isapprox(getFinalPoint(tInterval_1),[0.,0,0,0])
        tInterval_1.finalStep = true
        @test isapprox(getFinalPoint(tInterval_1),[.009,.0123847,2,0])

        tInterval_2 = TrackedInterval([-2.;1;;])
        tInterval_2.interval = [-.9;-.89;;]
        push!(tInterval_2.transforms,[.4;;4])
        tInterval_2.finalStep = true
        tInterval_2.preFinalTransforms = []
        push!(tInterval_2.preFinalTransforms,[1.;;0.])
        tInterval_2.finalInterval = [-5.;5;;]
        @test isapprox(getFinalPoint(tInterval_2),[3.8])
        tInterval_2.finalStep = false
        @test isapprox(getFinalPoint(tInterval_2),[0.])
    end
end

function test_contains()
    @testset "contains unit tests" begin
        # x in [-1,1], y in [-2,2]
        tInterval_1 = TrackedInterval([-1.;1.;;-2.;2.])

        # interior and the exact corners, which are inside a closed interval
        @test contains(tInterval_1, [0.0, 0.0])
        @test contains(tInterval_1, [-1.0, -2.0])
        @test contains(tInterval_1, [1.0, 2.0])
        @test contains(tInterval_1, [-1.0, 2.0])
        @test contains(tInterval_1, [0.5, -1.75])

        # outside in one dimension only, in each dimension and each direction. These are
        # the cases that a lexicographic comparison gets wrong: it stops at the first
        # coordinate that differs, so a point well inside on x was reported as contained
        # however far outside it was on y.
        @test contains(tInterval_1, [0.0, 5.0]) == false
        @test contains(tInterval_1, [0.0, -5.0]) == false
        @test contains(tInterval_1, [0.5, -9.0]) == false
        @test contains(tInterval_1, [2.0, 0.0]) == false
        @test contains(tInterval_1, [-2.0, 0.0]) == false

        # just outside, on each face
        @test contains(tInterval_1, [1.0 + 1e-12, 0.0]) == false
        @test contains(tInterval_1, [0.0, -2.0 - 1e-12]) == false

        # outside in every dimension at once
        @test contains(tInterval_1, [7.0, 7.0]) == false

        # contains reads the current interval, so it follows the interval as it moves
        tInterval_2 = TrackedInterval([-1.;1.;;-1.;1.])
        @test contains(tInterval_2, [0.9, 0.9])
        tInterval_2.interval = [-1. -1.; 0. 0.]
        @test contains(tInterval_2, [0.9, 0.9]) == false
        @test contains(tInterval_2, [-0.5, -0.5])

        # one dimension, and a degenerate interval that is a single point
        tInterval_3 = TrackedInterval([-3.;4.;;])
        @test contains(tInterval_3, [0.0])
        @test contains(tInterval_3, [-3.0])
        @test contains(tInterval_3, [4.0])
        @test contains(tInterval_3, [4.5]) == false

        tInterval_4 = TrackedInterval([2.;2.;;5.;5.])
        @test contains(tInterval_4, [2.0, 5.0])
        @test contains(tInterval_4, [2.0, 5.1]) == false
    end
end

function test_overlapsWith()
    @testset "overlapsWith unit tests" begin
        tInterval_1a = TrackedInterval([-1.;1;;-3;-2])
        tInterval_1b = TrackedInterval([-1.;1;;-1;1])
        @test overlapsWith(tInterval_1a,tInterval_1b) == false

        tInterval_2a = TrackedInterval([-3.;-2;;-1;1])
        tInterval_2b = TrackedInterval([-1.;1;;-1;1])
        @test overlapsWith(tInterval_2a,tInterval_2b) == false

        tInterval_3a = TrackedInterval([-4.;4;;-1;1;;-4;4])
        tInterval_3b = TrackedInterval([-1.;1;;-5;5;;-1;1])
        @test overlapsWith(tInterval_3a,tInterval_3b) == true
        
        tInterval_4a = TrackedInterval([-.54;-.54;;-.78;-.78])
        tInterval_4b = TrackedInterval([-.83;-.83;;.13;.13])
        @test overlapsWith(tInterval_4a,tInterval_4b) == false
    end
end

function test_startFinalStep()
    @testset "startFinalStep unit tests" begin
        tInterval_1 = TrackedInterval([1;2;;-1;1])
        @test tInterval_1.finalStep == false
        tInterval_1.transforms = [3;1;2;1;;4;1;2;3]
        startFinalStep(tInterval_1)
        @test tInterval_1.finalStep == true
        @test isapprox(tInterval_1.preFinalInterval,[1;2;;-1;1])
        @test isapprox(tInterval_1.preFinalTransforms,[3;1;2;1;;4;1;2;3])
    end
end