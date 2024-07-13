"""
x = pivot(A, b; ...)

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
function pivot(A,
               b::AbstractVector{T};
               tol::Float64=1e-8,
               max_iter=30*size(A,2)) where T


    # dimensions, initialize solution
    p,q = size(A)

    x = zeros(T, q) # primal variables
    y = -A'*b    # dual variables

    # parameters for swapping
    α = 3
    β = q+1

    # Store indices for the passive set, P
    #    we want Y[P] == 0, X[P] >= 0
    #    we want X[~P]== 0, Y[~P] >= 0
    P = BitArray(false for _ in 1:q)

    y[(!).(P)] =  A[:,(!).(P)]' * (A[:,P]*x[P] - b)

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
            x[P] =  A[:,P] \ b
		end
		y[(!).(P)] =  A[:,(!).(P)]' * ((A[:,P]*x[P]) - b)

        # check infeasibility
        @__dot__ V = (P & (x < -tol)) | (!P & (y < -tol))
        nV = sum(V)
    end

    x[(!).(P)] .= zero(eltype(x))
    return x
end


## if multiple right hand sides are provided, solve each problem separately.
function pivot(A,
               B::AbstractMatrix{T};
               use_parallel = true,
               kwargs...) where {T}

    n = size(A,2)
    k = size(B,2)

    # compute result for each column
    X = Array{T}(undef,n,k)
    if use_parallel && k > 1
        Threads.@threads for i = 1:k
            X[:,i] = pivot(A, B[:,i]; kwargs...)
        end
    else
        for i = 1:k
            X[:,i] = pivot(A, B[:,i]; kwargs...)
        end
    end

    return X
end
