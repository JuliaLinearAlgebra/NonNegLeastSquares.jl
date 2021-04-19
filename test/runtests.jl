using Test
using LinearAlgebra: norm
using NonNegLeastSquares: nonneg_lsq
using SparseArrays

#test specific
using Random
using PyCall
const pyopt = pyimport_conda("scipy.optimize", "scipy")

function test_case1()
    A = [ 0.53879488  0.65816267
          0.12873446  0.98669198
          0.24555042  0.00598804
          0.80491791  0.32793762 ]

    b = [0.888, 0.562, 0.255, 0.077]
    x = [0.15512102, 0.69328985] # approx solution from scipy
    return A, b, x
end

function test_case2()
    A = [ -0.24  -0.82   1.35   0.36   0.35
          -0.53  -0.20  -0.76   0.98  -0.54
           0.22   1.25  -1.60  -1.37  -1.94
          -0.51  -0.56  -0.08   0.96   0.46
           0.48  -2.25   0.38   0.06  -1.29 ]
    b = [-1.6, 0.19, 0.17, 0.31, -1.27]
    x = [2.2010416, 1.19009924, 0.0, 1.55001345, 0.0]
    return A, b, x
end

function test_case3() # non-float
    A = ones(Int, 4, 3)
    b = 2*ones(Int, 4)
    x = 2*ones(Int, 3)
    return A, b, x
end

function test_algorithm(fh::Function, ε::Real=1e-5)
    # Solve A*x = b for x, subject to x >=0
    A, b, x = test_case1()
    @test norm(fh(A,b) - x) < ε

    A, b, x = test_case2()
    @test norm(fh(A,b) - x) < ε

    # Test a bunch of random cases
    for i = 1:100
        m,n = rand(1:10),rand(1:10)
        A3 = randn(m,n)
        b3 = randn(m)
        x3,resid = pyopt.nnls(A3,b3)
        if resid > ε
            @test norm(fh(A3,b3) - x3) < ε
        else
            @test norm(A3*fh(A3,b3) - b3) < ε
        end
    end
end

nnls(A,b) = nonneg_lsq(A, b; alg=:nnls)
nnls_gram(A,b) = nonneg_lsq(A'*A, A'*b; alg=:nnls, gram=true)
fnnls(A,b) = nonneg_lsq(A, b; alg=:fnnls)
fnnls_gram(A,b) = nonneg_lsq(A'*A, A'*b; alg=:fnnls, gram=true)
pivot(A,b) = nonneg_lsq(A, b; alg=:pivot)
pivot_comb(A,b) = nonneg_lsq(A, b; alg=:pivot, variant=:comb)
pivot_cache(A,b) = nonneg_lsq(A, b; alg=:pivot, variant=:cache)

algs = [nnls, nnls_gram, fnnls, fnnls_gram, pivot, pivot_comb, pivot_cache]
errs = fill(1e-5, length(algs))

for (f, ε) in zip(algs, errs)
    print("testing ")
    @show f
    test_algorithm(f, ε)
    println("done")
end

#= non-float test fails, so revisit later
@testset "pivot_cache-non-float" begin
    A, b, x = test_case3()
    xi = pivot_cache(A, b)
    xf = pivot_cache(Float32.(A), Float32.(b))
    @test xi ≈ xf
end
=#

@testset "comb" begin
    A, b, x = test_case2()
    x0 = pivot_comb(A, b)
    P! = falses(size(A,2),1)
    x1 = nonneg_lsq(A, b; alg=:pivot, variant=:comb, P! = P!)
    @test x0 == x1
    @test P! != falses(size(A,2),1)
    @test all(x0[(!).(P!)] .== 0)
end

@testset "NNLS" begin include("nnls_test.jl") end
@testset "FNNLS" begin include("fnnls_test.jl") end
@testset "Pivot" begin include("pivot_test.jl") end
@testset "Sparse" begin include("sparse_test.jl") end
