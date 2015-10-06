module NonNegLeastSquares

export nonneg_lsq

## Algorithms
include("nnls.jl")
include("fnnls.jl")

## Common interface to algorithms
"""
x = nonneg_lsq(A, b; ...)

Computes the vector x that minimizes norm(b-A*x) subject to x >= 0, where A is
an (m x n) matrix and b is a (n x 1) column vector.

Optional arguments
------------------
    alg: a symbol specifying the algorithm to be used
    x0: initial guess for the solution
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
	else
		error("Specified algorithm :",alg," not recognized.")
	end
end

end