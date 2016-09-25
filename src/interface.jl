"""
**x = nonneg_lsq(A, B; ...)**
**x = nonneg_lsq(A'*A, A'*B; gram=true,...)**

Computes the (k-by-n) matrix X with that minimizes vecnorm(A*X - B) subject to
X >= 0, where A is an (m-by-k) matrix and B is a (m-by-n) matrix.

Optional arguments
------------------
**alg:** a symbol specifying the algorithm to be used

    :pivot - Block-pivoting active-set-like method (Kim & Park, 2011)
    :fnnls - Fast active-set method (Bro & De Jong, 1997)
    :nnls - Classic active-set method (Lawson & Hanson, 1974)
    :admm - Alternating Direction Method of Multipliers (e.g., Boyd et al., 2011)

**variant:** a symbol specifying the variant, if applicable, 

**gram:** a boolean indicating whether inputs A,B are Gram matrices A'*A,A'*B or data matrices A,B.

**tol:** tolerance for nonnegativity constraints

**max_iter:** maximum number of iterations before function gives up

"""
function nonneg_lsq(
        A::Matrix{Float64},
        B::Matrix{Float64};
        alg::Symbol = :pivot,
        variant::Symbol = :none,
        gram::Bool = false,
        kwargs...
    )

    # Check variant input
    if variant != :none
        if !(variant == :comb || variant == :cache)
            warn("Specified algorithm variant, :",variant," is not recognized.")
        elseif alg != :pivot
            warn("Algorithm variant, :",variant,", is not recognized for the specied algorithm, :",alg)
        end
    end

    if gram && alg == :nnls
        alg = :fnnls # fnnls is nnls using Gram matrices
    elseif gram && alg == :pivot
        alg = :pivot_cache # pivot_cache is pivot using Gram matrices
    elseif gram && !(alg in [:pivot_cache,:fnnls])
        error("Using the Gram interface is only allowed for the nnls, fnnls, and pivot algorithms.")
    end

    if alg == :nnls
        return nnls(A, B; kwargs...)
    elseif alg == :fnnls
        return fnnls(A, B; gram=gram, kwargs...)
    elseif alg == :pivot && variant == :cache
        return pivot_cache(A, B; gram=gram, kwargs...)
    elseif alg == :pivot && variant == :comb
        return pivot_comb(A, B; kwargs...)
    elseif alg == :pivot
        return pivot(A, B; kwargs...)
    elseif alg == :admm
        return admm(A, B; kwargs...)
    else
        error("Specified algorithm :",alg," not recognized.")
    end
end

# If second input is a vector, convert it to a matrix
nonneg_lsq(A::Matrix, b::Vector; kwargs...) = nonneg_lsq(A, b[:,:]; kwargs...)
