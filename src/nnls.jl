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
              max_iter=30*size(A,2),
              kwargs...)

    # dimensions, initialize solution
    m,n = size(A)
    x = zeros(n)

    # P is a bool array storing positive elements of x
    # i.e., x[P] > 0 and x[~P] == 0
    P = zeros(Bool,n)

    # We have reached an optimum when either:
    #   (a) all elements of x are positive (no nonneg constraints activated)
    #   (b) ∂f/∂x = A' * (b - A*x) > 0 for all nonpositive elements of x
    w = A' * (b - A*x)
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
        x = deepcopy(s)
        w = A' * (b - A*x)
    end
    return x
end

function nnls(A::Matrix{Float64},
              B::Matrix{Float64};
              use_parallel = true,
              kwargs...)

    m,n = size(A)
    k = size(B,2)

    if use_parallel && nprocs()>1
        X = SharedArray(Float64,n,k)
        @sync @parallel for i = 1:k
            X[:,i] = nnls(A, B[:,i]; kwargs...)
        end
        X = convert(Array,X)
    else
        X = Array(Float64,n,k)
        for i = 1:k
            X[:,i] = nnls(A, B[:,i]; kwargs...)
        end
    end

    return X
end
