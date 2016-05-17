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
function pivot(A::Matrix{Float64},
               b::Vector{Float64};
               tol::Float64=1e-8,
               max_iter=30*size(A,2))


    # dimensions, initialize solution
    p,q = size(A)

    x = zeros(q) # primal variables
    y = -A'*b    # dual variables

    # parameters for swapping
    α = 3
    β = q+1

    # Store indices for the passive set, P
    #    we want Y[P] == 0, X[P] >= 0
    #    we want X[~P]== 0, Y[~P] >= 0
    P = BitArray(q)

    x[P] =  A[:,P] \ b
    y[~P] =  A[:,~P]' * (A[:,P]*x[P] - b)

    # identify indices of infeasible variables
    V = (P & (x .< -tol)) | (~P & (y .< -tol))
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
		P = (P & ~V) | V & ~P

		# update primal/dual variables
		x[P] =  A[:,P] \ b
		y[~P] =  A[:,~P]' * ((A[:,P]*x[P]) - b)

        # check infeasibility
        V = (P & (x .< -tol)) | (~P & (y .< -tol))
        nV = sum(V)
    end

    x[~P] = 0.0
    return x
end


## if multiple right hand sides are provided, solve each problem separately.
function pivot(A::Matrix{Float64},
               B::Matrix{Float64};
               use_parallel = true,
               kwargs...)

    n = size(A,2)
    k = size(B,2)
    
    # compute result for each column
    if use_parallel && nprocs()>1
        X = SharedArray(Float64,n,k)
        @sync @parallel for i = 1:k
            X[:,i] = pivot(A, B[:,i]; kwargs...)
        end
        X = convert(Array,X)
    else
        X = Array(Float64,n,k)
        for i = 1:k
            X[:,i] = pivot(A, B[:,i]; kwargs...)
        end
    end

    return X
end

