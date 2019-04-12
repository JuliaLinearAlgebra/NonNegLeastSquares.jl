"""
x = pivot_cache(A, b; ...)

Solves non-negative least-squares problem by block principal pivoting method
(Algorithm 1) described in Kim & Park (2011).

Optional arguments:
    tol: tolerance for nonnegativity constraints
    max_iter: maximum number of iterations

References:
	J. Kim and H. Park, Fast nonnegative matrix factorization: An
	active-set-like method and comparisons, SIAM J. Sci. Comput., 33 (2011),
	pp. 3261–3281.
"""
function pivot_cache(AtA,
                    Atb::AbstractVector{T};
                    tol::Float64=1e-8,
                    max_iter=30*size(AtA,2)) where {T}


    # dimensions, initialize solution
    q = size(AtA,1)

    x = zeros(T, q) # primal variables
    y = -Atb    # dual variables

    # parameters for swapping
    α = 3
    β = q+1

    # Store indices for the passive set, P
    #    we want Y[P] == 0, X[P] >= 0
    #    we want X[~P]== 0, Y[~P] >= 0
    P = BitArray(false for _ in 1:q)

    y[(!).(P)] = AtA[(!).(P),P]*x[P] - Atb[(!).(P)]

    # identify indices of infeasible variables
    V = @__dot__ (P & (x < -tol)) | (!P & (y < -tol))
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
		@__dot__ P = (P & !V) | (V & !P)

		# update primal/dual variables
		if !all(!, P)
        	x[P] = _get_primal_dual(AtA, Atb, P)
		end
        #x[(!).(P)] = 0.0
        y[(!).(P)] = AtA[(!).(P),P]*x[P] - Atb[(!).(P)]
        #y[P] = 0.0

        # check infeasibility
        @__dot__ V = (P & (x < -tol)) | (!P & (y < -tol))
        nV = sum(V)
    end

    x[(!).(P)] .= zero(eltype(x))
    return x
end

@inline function _get_primal_dual(AtA::SparseArrays.SparseMatrixCSC, Atb, P)
	return qr(AtA[P,P])\Atb[P]
end
@inline function _get_primal_dual(AtA, Atb, P)
	return pinv(AtA[P,P])*Atb[P]
end


## if multiple right hand sides are provided, solve each problem separately.
function pivot_cache(A,
                     B::AbstractMatrix{T};
                     gram::Bool = false,
                     use_parallel::Bool = false,
                     kwargs...) where {T}

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
        X = SharedArray{T}(n,k)
        @sync @distributed for i = 1:k
            X[:,i] = pivot_cache(AtA, AtB[:,i]; kwargs...)
        end
        X = convert(Array,X)
    else
        X = Array{T}(undef,n,k)
        for i = 1:k
            X[:,i] = pivot_cache(AtA, AtB[:,i]; kwargs...)
        end
    end

    return X
end
