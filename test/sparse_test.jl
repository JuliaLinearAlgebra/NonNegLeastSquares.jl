# test simple solving
Random.seed!(1234)
sA = sprand(20, 20, 0.1)
b = rand(20)
A = collect(sA)

@test fnnls(sA, b) ≈ fnnls(A, b)
@test nnls(sA, b) ≈ nnls(A, b)
@test pivot(sA, b) ≈ pivot(A, b)
@test nonneg_lsq(sA,b;alg=:pivot, variant=:cache) ≈ nonneg_lsq(A,b;alg=:pivot, variant=:cache)
