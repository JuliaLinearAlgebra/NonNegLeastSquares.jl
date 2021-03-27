using NonNegLeastSquares

## compare two algorithms
function compare(alg1,alg2;var1=:none,var2=:none,m=200,n=200,k=3)
	println("\nComparing $alg1:$var1 to $alg2:$var2 with A = randn($m,$k) and B = randn($m,$n)")
	println("-------------------------------------------------------------------------------------")

	A,B = randn(m,k),randn(m,n)

	print(uppercase(string(alg1)),":",var1," → ")
	@time X1 = nonneg_lsq(A,B,alg=alg1,variant=var1)

	print(uppercase(string(alg2)),":",var2," → ")
	@time X2 = nonneg_lsq(A,B,alg=alg2,variant=var2)

	if norm(X1-X2)/sqrt(length(X1)) > 1e-10
		warn("algorithms did not converge on the same answer.")
	end

end

compare(:nnls, :fnnls; m=100, n=1, k=100);

compare(:fnnls, :pivot; m=1000, n=1, k=1000);

compare(:fnnls, :admm; m=1000, n=1, k=1000);

compare(:pivot, :admm; m=1000, n=1, k=1000);

compare(:fnnls, :pivot; m=1000, n=1000, k=3);

compare(:pivot, :pivot; var2=:cache, m=100, n=100, k=100);

compare(:pivot, :pivot; var2=:comb, m=100, n=100, k=100);

compare(:pivot, :pivot; var1=:cache, var2=:comb, m=100, n=100, k=100);

compare(:pivot, :pivot; var1=:none, var2=:comb, m=5000, n=5000, k=3);

compare(:pivot, :pivot; var1=:cache, var2=:comb, m=10000, n=10000, k=10);
