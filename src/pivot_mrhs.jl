"""
x = pivot_mrhs(A, b; ...)

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
function pivot_mrhs(A::Matrix{Float64},
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

    # Divide elements into two sets
    F = BitArray(zeros(q,r)) # we want Y[F] == 0, X[F] >= 0
    G = BitArray(ones(q,r))  # we want X[G] == 0, Y[G] >= 0

    # Update primal and dual variables
    cssls!(AtA,AtB,X,F) # overwrite X[F]
    Y = AtA*X - AtB

    # identify infeasible columns of X
    V = (F & (X .< -tol)) | (G & (Y .< -tol)) # infeasible variables
    infeasible_cols = any(V, 1) # collapse each column

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
        
        # Update partition of variables in F and G
        #     F & ~V removes infeasible variables from F 
        #     V & G  moves infeasible variables in G to F
        F = (F & ~V) | V & G
        G = ~F # G is always the complement of set F

        # Update primal and dual variables
        cssls!(AtA,AtB,X,F) # overwrite X[F]
        X[G] = 0.0
        Y[:,infeasible_cols] = AtA*X[:,infeasible_cols] - AtB[:,infeasible_cols]
        Y[F] = 0.0 

        # identify infeasible columns of X
        V = (F & (X .< -tol)) | (G & (Y .< -tol)) # infeasible variables
        infeasible_cols = any(V, 1) # collapse each column
    end 

    X[G] = 0.0
    return X
end
