using ArrayViews

"""
x = admm(A, B; ...)

Solves non-negative least-squares problem by the Alternating Direction Method
of Multipliers (ADMM).

Optional arguments:
    tol: tolerance for nonnegativity constraints
    max_iter: maximum number of iterations

References:
    S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein (2011). Distributed
    Optimization and Statistical Learning via the Alternating Direction Method
    of Multipliers. Foundations and Trends in Machine Learning.

    http://stanford.edu/~eryu/nnlsqr.html
"""
function admm(A::Matrix{Float64},
	          B::Matrix{Float64};
	          ρ=vecnorm(A)^2/size(A,2),
	          kwargs...)

	println(ρ)

	# Dimensions
	m,k = size(A)
	n = size(B,2)

	# Cache matrices
	AtB = A'*B 
	AtAρ = A'*A + ρ*eye(k)
	#AtAρ = A'*A # A'*A + eye(k)*ρ
	#for i = 1:k
	#	AtAρ[i,i] += ρ
	#end

	# Cache cholesky factorization
	L = cholfact(AtAρ)
	
	# Matrix storing the solutions
	X = zeros(k,n)

	for i = 1:n
		# Select column of B to solve
		Atb = view(AtB,:,i)
		x,z,u = zeros(k),zeros(k),zeros(k)
		
		x = L \ (Atb+ρ*(z-u))
		z = max(0,x+u)
		u = u+x-z
		
		# Solve
		#while (sum(abs(z-x))/k) > 1e12
		for j = 1:1000
			x = L \ (Atb+ρ*(z-u))
			z = max(0,x+u)
			u = u+x-z
		end

		# z ≈ x is the solution to A*x = B[:,i]
		X[:,i] = x
	end
	
	return X
end

