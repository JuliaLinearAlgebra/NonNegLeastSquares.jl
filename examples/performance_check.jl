using NonNegLeastSquares

## compare two algorithms
function compare(alg1,alg2;m=200,n=200,k=3)
	println("\nComparing :$alg1 to :$(alg2), A = randn($m,$k) and B = randn($m,$n)")
	println("-------------------------------------------------------------------")

	A,B = randn(m,k),randn(m,n)

	print(uppercase(string(alg1))," → ")
	@time X1 = nonneg_lsq(A,B,alg=alg1)

	print(uppercase(string(alg2))," → ")
	@time X2 = nonneg_lsq(A,B,alg=alg2)

	if vecnorm(X1-X2) > 0.1
		warn("algorithms did not converge on the same answer.")
	end

end

A = randn(10,10)
B = randn(10,1)
nonneg_lsq(A,B,alg=:nnls)
nonneg_lsq(A,B,alg=:fnnls)
nonneg_lsq(A,B,alg=:pivot_mrhs)
nonneg_lsq(A,B,alg=:pivot_srhs)

compare(:nnls,:fnnls;m=100,n=1,k=100);

compare(:fnnls,:pivot_srhs;m=1000,n=1000,k=3);

compare(:pivot_srhs,:pivot_mrhs;m=100,n=100,k=100);

compare(:pivot_srhs,:pivot_mrhs;m=5000,n=5000,k=3);

