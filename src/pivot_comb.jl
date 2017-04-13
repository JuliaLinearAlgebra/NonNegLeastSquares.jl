"""
x = pivot_comb(A, b; ...)

Solves non-negative least-squares problem by block principal pivoting method 
with combinatorial grouping of least-squares problems. Algorithm 2 described
in Kim & Park (2011).

Optional arguments:
    tol: tolerance for nonnegativity constraints
    max_iter: maximum number of iterations

References:
    J. Kim and H. Park, Fast nonnegative matrix factorization: An
    active-set-like method and comparisons, SIAM J. Sci. Comput., 33 (2011),
    pp. 3261–3281.
"""
function pivot_comb(A::Matrix{Float64},
               B::Matrix{Float64};
               tol::Float64=1e-8,
               max_iter=30*size(A,2))

    # precompute constant portion of pseudoinverse
    AtA = A'*A
    AtB = A'*B

    # dimensions, initialize solution
    q,r = size(AtB)
    X = zeros(q,r) # primal variables
    Y = -AtB       # dual variables

    # parameters for swapping
    α = ones(r)*3
    β = ones(r)*(q+1)

    # Store indices for the passive set, P
    #    we want Y[P] == 0, X[P] >= 0
    #    we want X[.~P]== 0, Y[.~P] >= 0
    P = zeros(Bool,q,r)

    # Update primal and dual variables
    cssls!(AtA,AtB,X,P) # overwrite X[P]
    Y = AtA*X - AtB

    # identify infeasible columns of X
    infeasible_cols = Array{Bool}(size(X,2))
    
    V = (P .& (X .< -tol)) .| (.~P .& (Y .< -tol)) # infeasible variables
    any!(infeasible_cols, V') # collapse each column

    # while infeasible
    while any(infeasible_cols)

        # check progress
        for j = 1:r
            nV = sum(V[:,j])
            if nV < β[j]
                # infeasible variables decreased for column j
                β[j] = nV  # store number of infeasible variables
                α[j] = 3   # reset α
            else
                # infeasible variables stayed the same or increased
                if α[j] >= 1
                    α[j] = α[j]-1 # tolerate increases for α cycles
                else
                    # backup rule
                    i = findlast(V[:,j])
                    V[:,j] = zeros(Bool,q)
                    V[i] = true
                end
            end
        end
        
        # Update passive set
        #     P & .~V removes infeasible variables from P 
        #     V & .~P moves infeasible variables to the
        P = (P .& .~V) .| (V .& .~P)

        # Update primal and dual variables
        cssls!(AtA,AtB,X,P) # overwrite X[P]
        X[.~P] = 0.0
        Y[:,infeasible_cols] = AtA*X[:,infeasible_cols] - AtB[:,infeasible_cols]
        Y[P] = 0.0 

        # identify infeasible columns of X
        V = (P .& (X .< -tol)) .| (.~P .& (Y .< -tol)) # infeasible variables
        any!(infeasible_cols, V') # collapse each column
    end 

    X[.~P] = 0.0
    return X
end
