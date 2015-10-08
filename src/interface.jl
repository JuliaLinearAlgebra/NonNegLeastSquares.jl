"""
x = nonneg_lsq(A, b; ...)

Computes the vector x that minimizes norm(b-A*x) subject to x >= 0, where A is
an (m x n) matrix and b is a (n x 1) column vector.

Optional arguments
------------------
    alg: a symbol specifying the algorithm to be used
    tol: tolerance for nonnegativity constraints
    max_iter: maximum number of iterations
"""
function nonneg_lsq(
	A::Matrix{Float64},
	b::Vector{Float64};
	alg::Symbol = :pivot,
	kwargs...
    )

	if alg == :nnls
		return nnls(A, b; kwargs...)
	elseif alg == :fnnls
		return fnnls(A'*A, A'*b; kwargs...)
	elseif alg == :convex
		return convex_nnls(A, b; kwargs...)
	elseif alg == :pivot
		return pivot_nnls(A, b; kwargs...)
	else
		error("Specified algorithm :",alg," not recognized.")
	end
end


"""
X = nonneg_lsq(A, B; ...)

Computes the (n x k) matrix X that minimizes norm(B-A*X) subject to X >= 0,
where A is an (m x n) matrix and B is a (n x k) matrix.

Optional arguments
------------------
    alg: a symbol specifying the algorithm to be used
    tol: tolerance for nonnegativity constraints
    max_iter: maximum number of iterations
"""
function nonneg_lsq(
	A::Matrix{Float64},
	B::Matrix{Float64};
	alg::Symbol = :pivot,
	kwargs...
	)

	# check dimensions
	m,n = size(A)
	k = size(B,2)
	if size(B,1) != n
		error("Dimension mismatch")
	end

	# List of algorithms that only work with vector "b"
	single_rhs_algs = [:nnls,:fnnls,:pivot]

	if alg in single_rhs_algs
		# Solve each column of X sequentially
		X = zeros(n,k)
		for i = 1:k
			X[:,i] = nonneg_lsq(A, B[:,i]; alg=alg, kwargs...)
		end
		return X
	elseif alg == :convex
		return convex_nnls(A, b; kwargs...)
	else
		error("Specified algorithm :",alg," not recognized.")
	end
end
