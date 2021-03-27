"""
x = coord_desc(A, B; ...)

Solves non-negative least-squares problem by coordinate descent (CD).

Optional arguments:
    ε: tolerance for stopping (small number times sqrt[m*n] by default)

References:
    Reference has been lost...
"""

function coord_desc(A, B; ε = sqrt(size(A,2)*size(B,2))*1e-15, max_iter = 500)
    rows,vars = size(A)
    ATA,ATY = transpose(A) * A, transpose(A) * B
    a = zeros(vars)
    μ = -ATY
    Haf = similar(aTB)
    @inbounds for iter in 1:max_iters
        Haf .= ATA * a .- ATB
        all(>=(-ε), Haf) && break
        for v in 1:vars
            initial = a[v]
            a[v,1]  = max(a[v] - μ[v] / ATA[v,v], 0.0)
            ∇       = x[v] - initial
            μ      .+= ∇ .* @views ATA[:,v]
        end
    end
    return a
end


# variables = 100
# X    = randn(200, variables)
# beta = rand(variables, 1)
# Y    = X * beta
# r = NNLS_CD(X, Y; ϵ=1e-9, max_iters=300)
# sum(abs,beta .- r)