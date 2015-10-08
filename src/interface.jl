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
	alg::Symbol = :fnnls,
	kwargs...
    )

	if alg == :nnls
		return nnls(A, b; kwargs...)
	elseif alg == :fnnls
		return fnnls(A'*A, A'*b; kwargs...)
	elseif alg == :convex
		return convex_nnls(A, b; kwargs...)
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
	alg::Symbol = :fnnls,
	kwargs...
	)

	# check dimensions
	m,n = size(A)
	if size(B,1) != n
		error("Dimension mismatch")
	end

	if alg == :convex
		# Convex.jl handles matrix variables
		return convex_nnls(A, B; kwargs...)
	elseif (alg == :nnls) || (alg == :fnnls)
		# NNLS and Fast NNLS solve each column sequentially
		k = size(B,2)
		X = zeros(n,k)
		for i = 1:k
			X[:,i] = nonneg_lsq(A,B[:,i];alg=alg,kwargs...)
		end
		return X
	else
		error("Specified algorithm :",alg," not recognized.")
	end
end
