"""
x = nnls(A, b; ...)

Solves non-negative least-squares problem by the active set method
of Lawson & Hanson (1974).

Optional arguments:
    tol: tolerance for nonnegativity constraints
    max_iter: maximum number of iterations (counts inner loop iterations)

References:
    Lawson, C.L. and R.J. Hanson, Solving Least-Squares Problems,
    Prentice-Hall, Chapter 23, p. 161, 1974.
"""
function nnls(A::Matrix{Float64},
              b::Vector{Float64};
              tol::Float64=1e-8,
              max_iter=5*size(A,2))

    m,n = size(A)

    # check x0 input
    neg_entries = x0 .< 0.0
    if any(neg_entries)
        warn("x0 contains negative values, projecting onto positive orthant")
        x0[neg_entries] = 0.0
    end
    x = x0 # rename x0, initialize solution
    
    # P is a bool array storing positive elements of x
    # i.e., x[P] > 0 and x[~P] == 0
    P = x .> tol 
    w = A' * (b - A*x)

    # KKT conditions, we have reached an optimum when either:
    #   (a) all elements of x are positive (no nonneg constraints activated)
    #   (b) A' * (b - A*xᵢ) > 0 for all nonpositive elements xᵢ
    iter = 0
    while sum(P)<n && any(w[~P].>tol) && iter < max_iter

        # find i that maximizes w, restricting i to indices not in P
        # Note: the while loop condition guarantees at least one w[~P]>0
        i = indmax(w .* ~P) 

        # Move i to P
        P[i] = true

        # Solve least-squares problem, with zeros for columns/elements not in P
        Ap = zeros(m,n)
        Ap[:,P] = A[:,P]
        s = pinv(Ap)*b
        s[~P] = 0.0 # zero out elements not in P

        # Inner loop: deal with negative elements of s
        while any(s[P].<=tol)
            iter += 1

            # find indices in P where s is negative
            ind = (s.<=tol) & P

            # calculate step size, α, to prevent any xᵢ from going negative
            α = minimum(x[ind] ./ (x[ind] - s[ind]))

            # update solution (pushes some xᵢ to zero)
            x += α*(s-x)

            # Remove all i in P where x[i] == 0
            for i = 1:n
                if P[i] && abs(x[i]) < tol
                    # remove i from P
                    P[i] = false 
                    # zero out column i of Ap
                    Ap[:,i] *= 0.0
                end
            end

            # Solve least-squares problem again, zeroing nonpositive columns
            s = pinv(Ap)*b
            s[~P] = 0.0 # zero out elements not in P
        end

        # update solution
        x = s
        w = A' * (b - A*x)
    end
    return x
end
