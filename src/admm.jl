"""
x = admm(A, B; ...)

Solves non-negative least-squares problem by the Alternating Direction Method
of Multipliers (ADMM).

Optional arguments:
    ρ: penalty parameter (set heuristically by default)
    ε: tolerance for stopping (small number times sqrt[m*n] by default)

References:
    S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein (2011). Distributed
    Optimization and Statistical Learning via the Alternating Direction Method
    of Multipliers. Foundations and Trends in Machine Learning.

    http://stanford.edu/~eryu/nnlsqr.html
"""
function admm(A::Matrix{Float64},
              B::Matrix{Float64};
              ρ=max(1.0, vecnorm(A)^2/size(A,2)),
              ε=sqrt(size(A,2)*size(B,2))*1e-15,
              max_iter=1000*size(A,2),
              kwargs...)

    # Dimensions
    m,k = size(A)
    n = size(B,2)

    # Cache matrices
    AtB = A'*B

    # Cache cholesky factorization
    L = cholfact(A'*A + ρ*I)

    # Matrix storing the solutions
    X = zeros(k,n)

    # Initialize variables
    Z,U = zeros(k,n),zeros(k,n)
    X = L \ (AtB+ρ*(Z-U))

    # Solve
    err = Inf
    for i = 1:max_iter
        Z = max(0,X+U)
        U = U+X-Z
        X = L \ (AtB+ρ*(Z-U))
        vecnorm(X-Z) < ε && break
    end

    return max(0, X)
end

