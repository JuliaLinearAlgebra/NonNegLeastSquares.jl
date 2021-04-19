"""
    x = pivot_cache(A, b; ...)

Solve non-negative least-squares problem by block principal pivoting method
(Algorithm 1) described in Kim & Park (2011).

Optional arguments:
* `tol` tolerance for nonnegativity constraints
 default `10^floor(log10(eps(T)^0.5))` which is `1e-8` for `Float64`
* `max_iter` maximum number of iterations, default `30 * size(A,2)`

References:
    J. Kim and H. Park, Fast nonnegative matrix factorization: An
    active-set-like method and comparisons, SIAM J. Sci. Comput., 33 (2011),
    pp. 3261–3281.
"""
function pivot_cache(
    AtA,
    Atb::AbstractVector{T};
    tol::Real = 10^floor(log10(eps(T)^0.5)),
    max_iter=30 * size(AtA,2),
) where {T <: AbstractFloat}

    # dimensions, initialize solution
    q = size(AtA,1)

    x = zeros(T, q) # primal variables
    y = -Atb # dual variables

    # parameters for swapping
    α = 3
    β = q+1

    # Store indices for the passive set, P
    #    we want Y[P] == 0, X[P] >= 0
    #    we want X[~P]== 0, Y[~P] >= 0
    P = falses(q)

    y[(!).(P)] = AtA[(!).(P),P] * x[P] - Atb[(!).(P)]

    # identify indices of infeasible variables
    V = @. (P & (x < -tol)) | (!P & (y < -tol))
    nV = sum(V)

    # while infeasible (number of infeasible variables > 0)
    while nV > 0

        if nV < β
            # infeasible variables decreased
            β = nV  # store number of infeasible variables
            α = 3   # reset α
        else
            # infeasible variables stayed the same or increased
            if α >= 1
                α = α-1 # tolerate increases for α cycles
            else
                # backup rule
                i = findlast(V)
                V = zeros(Bool,q)
                V[i] = true
            end
        end

        # update passive set
        #     P & ~V removes infeasible variables from P
        #     V & ~P  moves infeasible variables in ~P to P
        @. P = (P & !V) | (V & !P)

        # update primal/dual variables
        if !all(!, P)
            x[P] = _get_primal_dual(AtA, Atb, P)
        end
        #x[(!).(P)] = 0.0
        y[(!).(P)] = AtA[(!).(P),P]*x[P] - Atb[(!).(P)]
        #y[P] = 0.0

        # check infeasibility
        @. V = (P & (x < -tol)) | (!P & (y < -tol))
        nV = sum(V)
    end

    x[(!).(P)] .= zero(eltype(x))
    return x
end

@inline function _get_primal_dual(AtA::SparseArrays.SparseMatrixCSC, Atb, P)
    if VERSION < v"1.2"
        return pinv(Array(AtA[P,P]))*Atb[P]
    else
        return qr(AtA[P,P]) \ Atb[P]
    end
end

@inline function _get_primal_dual(AtA, Atb, P)
    return pinv(AtA[P,P])*Atb[P]
end


# if multiple right hand sides are provided, solve each problem separately.
function pivot_cache(
    A,
    B::AbstractMatrix{T};
    gram::Bool = false,
    use_parallel::Bool = true,
    kwargs...
) where {T <: AbstractFloat}

    n = size(A,2)
    k = size(B,2)

    if gram
        # A,B are actually Gram matrices
        AtA = A
        AtB = B
    else
        # cache matrix computations
        AtA = A'*A
        AtB = A'*B
    end

    # compute result for each column
    if use_parallel && nprocs()>1
        X = @distributed (hcat) for i = 1:k
            pivot_cache(AtA, AtB[:,i]; kwargs...)
        end
    else
        X = Array{T}(undef,n,k)
        for i = 1:k
            X[:,i] = pivot_cache(AtA, AtB[:,i]; kwargs...)
        end
    end

    return X
end

# for non-float types, promote to Float32 to ensure sensible tol
function pivot_cache(A, B; kwargs...)
    T = promote_type(eltype(A), eltype(B), Float32)
    pivot_cache(T.(A), T.(B); kwargs...)
end
