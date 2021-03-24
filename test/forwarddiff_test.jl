# test simple solving


Random.seed!(1234)

b = rand(20)
A = rand(20, 20)

dual=ForwardDiff.Dual{:t}(1.0,one(1.))
db = copy(b).*dual
dA = copy(A).*dual

matrixes = [A, dA]
bvecs = [b, db]
for bv in bvecs, Am in matrixes
    @test fnnls(Am, bv) ≈ fnnls(A, b)
    @test pivot(Am, bv) ≈ pivot(A, b)
end
@test nnls(dA, db) ≈ nnls(A, b)
@test_broken nnls(dA,b) ≈ nnls(A, b)
@test_broken nnls(A,db) ≈ nnls(A, b)
@test_broken nonneg_lsq(dA,db;alg=:pivot, variant=:cache) ≈ nonneg_lsq(A,b;alg=:pivot, variant=:cache)
@test_broken nonneg_lsq(dA,db;alg=:pivot, variant=:comb) ≈ nonneg_lsq(A,b;alg=:pivot, variant=:comb)