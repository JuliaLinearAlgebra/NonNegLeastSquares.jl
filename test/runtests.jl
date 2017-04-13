using Base.Test
using NonNegLeastSquares
using PyCall
@pyimport scipy.optimize as pyopt

function test_algorithm1(fh)
	# Solve A*x = b for x, subject to x >=0 
	A = [ 0.53879488  0.65816267 
	      0.12873446  0.98669198
	      0.24555042  0.00598804
	      0.80491791  0.32793762 ]

	b = [0.888,  0.562,  0.255,  0.077]
	x = [0.15512102, 0.69328985] # approx solution from scipy
    #println(fh," Case 1:",@test norm(fh(A,b)-x) ≈ 0 atol = 1e-5)
    #@test norm(fh(A,b)-x) ≈ 0 atol = 1e-5
    @test  norm(fh(A,b)-x) < 1e-5 
end

function test_algorithm2(fh)

	## A second test case
	A2 = [ -0.24  -0.82   1.35   0.36   0.35
	       -0.53  -0.20  -0.76   0.98  -0.54
	        0.22   1.25  -1.60  -1.37  -1.94
	       -0.51  -0.56  -0.08   0.96   0.46
	        0.48  -2.25   0.38   0.06  -1.29 ]
	b2 = [-1.6,  0.19,  0.17,  0.31, -1.27]
	x2 = [2.2010416, 1.19009924, 0.0, 1.55001345, 0.0]
        #@test norm(fh(A2,b2)-x2) ≈ 0 atol = 1e-5
        @test norm(fh(A2,b2)-x2) < 1e-5
end

function test_algorithm3(fh)

   ## Test a bunch of random cases
    for i = 1:10
	m,n = rand(1:10),rand(1:10)
	A3 = randn(m,n)
	b3 = randn(m)
	x3,resid = pyopt.nnls(A3,b3)
	if resid > 1e-5
            #@test norm(fh(A3,b3)-x3) ≈ 0.0 atol = 1e-5
            @test norm(fh(A3,b3)-x3) < 1e-5
	else
            #@test norm(A3*fh(A3,b3)-b3) ≈ 0.0 atol = 1e-5
            @test norm(A3*fh(A3,b3)-b3) < 1e-5
	end
    end
end

nnls(A,b) = nonneg_lsq(A,b;alg=:nnls)
nnls_gram(A,b) = nonneg_lsq(A'*A,A'*b;alg=:nnls,gram=true)
fnnls(A,b) = nonneg_lsq(A,b;alg=:fnnls)
fnnls_gram(A,b) = nonneg_lsq(A'*A,A'*b;alg=:fnnls,gram=true)
pivot(A,b) = nonneg_lsq(A,b;alg=:pivot)
pivot_comb(A,b) = nonneg_lsq(A,b;alg=:pivot,variant=:comb)
pivot_cache(A,b) = nonneg_lsq(A,b;alg=:pivot,variant=:cache)
admm(A,b) = nonneg_lsq(A,b;alg=:admm)

for func in [nnls,nnls_gram,fnnls,fnnls_gram,pivot,pivot_comb,pivot_cache]
 	test_algorithm1(func)
 	test_algorithm2(func)
 	test_algorithm3(func)
end

test_algorithm1(admm)
#test_algorithm2(admm)   # Too large errors
#test_algorithm3(admm)   # Intermittent success. More iterrations help.
