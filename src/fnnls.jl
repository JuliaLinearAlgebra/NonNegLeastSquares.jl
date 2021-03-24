"""
    x = fnnls(AtA, Atb; ...)

Returns `x` that solves `A*x = b` in the least-squares sense, subject to `x >=0`.
The inputs are the cross-products `AtA = A'*A` and `Atb = A'*b`.
Uses the modified active set method of Bro and De Jong (1997).

Optional arguments:
* `tol`: tolerance for nonnegativity constraints
* `max_iter`: maximum number of iterations (counts inner loop iterations)

References:
    Bro R, De Jong S. A fast non-negativitity-constrained least squares
    algorithm. Journal of Chemometrics. 11, 393–401 (1997)
"""
function fnnls(AtA,
               Atb::AbstractVector{T};
               tol::Float64=1e-8,
               max_iter=30*size(AtA,2)) where {T}

    XT=typeof(oneunit(T)/oneunit(eltype(AtA)))
    n = size(AtA,1)
    x = zeros(XT, n)
    s = zeros(XT, n)

    # P is a bool array storing positive elements of x
    # i.e., x[P] > 0 and x[~P] == 0
    P = x .> tol
    w = Atb - AtA*x

    # We have reached an optimum when either:
    #   (a) all elements of x are positive (no nonneg constraints activated)
    #   (b) ∂f/∂x = A' * (b - A*x) > 0 for all nonpositive elements of x
    iter = 0
    while sum(P)<n && any(w[(!).(P)] .> tol) && iter < max_iter

        # find i that maximizes w, restricting i to indices not in P
        # Note: the while loop condition guarantees at least one w[~P]>0
        i = argmax(w .* (!).(P))

        # Move i to P
        P[i] = true

        # Solve least-squares problem, with zeros for columns/elements not in P
        s[P] = AtA[P,P] \ Atb[P]
        s[(!).(P)] .= zero(eltype(s)) # zero out elements not in P

        # Inner loop: deal with negative elements of s
        while any(s[P].<=tol)
            iter += 1

            # find indices in P where s is negative
            ind = @__dot__ (s <= tol) & P

            # calculate step size, α, to prevent any xᵢ from going negative
            α = minimum(x[ind] ./ (x[ind] - s[ind]))

            # update solution (pushes some xᵢ to zero)
            x += α*(s-x)

            # Remove all i in P where x[i] == 0
            for i = 1:n
                if P[i] && abs(x[i]) < tol
                    P[i] = false # remove i from P
                end
            end

            # Solve least-squares problem again, zeroing nonpositive columns
            s[P] = AtA[P,P] \ Atb[P]
            s[(!).(P)] .= zero(eltype(s)) # zero out elements not in P
        end

        # update solution
        x = deepcopy(s)
        w .= Atb - AtA*x
    end
    return x
end

function fnnls(A,
               B::AbstractMatrix;
               gram::Bool = false,
               use_parallel::Bool = true,
               kwargs...)

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

    if use_parallel && nprocs()>1
        X = @distributed (hcat) for i=1:k
            fnnls(AtA, AtB[:,i]; kwargs...)
        end
    else
        X = Array{eltype(B)}(undef,n,k)
        for i = 1:k
            X[:,i] = fnnls(AtA, AtB[:,i]; kwargs...)
        end
    end

    return X
end
