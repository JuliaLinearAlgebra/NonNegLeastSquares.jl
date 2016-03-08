"""
x = nnls_gram(A, b; ...)

Solves non-negative least-squares problem by the active set method
of Lawson & Hanson (1974), using Gram matrices instead of data matrices.

Optional arguments:
    tol: tolerance for nonnegativity constraints
    max_iter: maximum number of iterations (counts inner loop iterations)

References:
    Lawson, C.L. and R.J. Hanson, Solving Least-Squares Problems,
    Prentice-Hall, Chapter 23, p. 161, 1974.
"""
function nnls_gram(AA::Matrix{Float64},
              Ab::Vector{Float64};
              tol::Float64=1e-8,
              max_iter=30*size(AA,2))

    # dimensions, initialize solution
    n = size(AA,1)
    x = zeros(n)

    # P is a bool array storing positive elements of x
    # i.e., x[P] > 0 and x[~P] == 0
    P = zeros(Bool,n)

    # We have reached an optimum when either:
    #   (a) all elements of x are positive (no nonneg constraints activated)
    #   (b) ∂f/∂x = A' * (b - A*x) > 0 for all nonpositive elements of x
    w = Ab - AA*x
    iter = 0
    while sum(P)<n && any(w[~P].>tol) && iter < max_iter

        # find i that maximizes w, restricting i to indices not in P
        # Note: the while loop condition guarantees at least one w[~P]>0
        i = indmax(w .* ~P) 

        # Move i to P
        P[i] = true

        # Solve least-squares problem, with zeros for columns/elements not in P
				ApAp = AA[P,P]
				Abp = Ab[P,:]
				s = zeros(n)
				s[P] = ApAp \ Abp

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
										ApAp[i,:] = 0.0
										ApAp[:,i] = 0.0
										Abp[i,:] = 0.0
                end
            end

            # Solve least-squares problem again, zeroing nonpositive columns
						s = ApAp \ Abp
            s[~P] = 0.0 # zero out elements not in P
        end

        # update solution
        x = deepcopy(s)
        w = Ab - AA*x
    end
    return x
end

function nnls_gram(AA::Matrix{Float64},
              AB::Matrix{Float64};
              kwargs...)

    n = size(AA,1)
    k = size(AB,2)

    X = zeros(n,k)
    for i = 1:k
        X[:,i] = nnls_gram(AA, AB[:,i]; kwargs...)
    end
    return X
end
