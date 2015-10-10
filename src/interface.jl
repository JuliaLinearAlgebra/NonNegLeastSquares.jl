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
	B::Matrix{Float64};
	alg::Symbol = :pivot,
	kwargs...
    )

	if alg == :nnls
		return nnls(A, B; kwargs...)
	elseif alg == :fnnls
		return fnnls(A, B; kwargs...)
	elseif alg == :convex
		return convex_nnls(A, B; kwargs...)
	elseif alg == :pivot || alg == :pivot_srhs
		return pivot_srhs(A, B; kwargs...)
	elseif alg == :pivot_mrhs
		return pivot_mrhs(A, B; kwargs...)
	else
		error("Specified algorithm :",alg," not recognized.")
	end
end

# If second input is a vector, convert it to a matrix
nonneg_lsq(A::Matrix, b::Vector; kwargs...) = nonneg_lsq(A, b[:,:]; kwargs...)
