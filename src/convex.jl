using Convex
using SCS
using ECOS

"""
x = convex_nnls(A, b, solver; kwargs...)

Solves the nonnegative least-squares problem using Convex.jl and the solver
specified by "solver". 
"""
function convex_nnls(
	A::Matrix{Float64},
	B::Array{Float64};
	solver::Symbol = :SCS,
	kwargs...)

	# Check dimensions
	if ndims(B) > 2
		error("B must be a matrix or a vector")
	elseif ndims(B) == 2
		X = Variable(size(B,1),size(B,2))
	else
		X = Variable(length(B))
	end

	# declare problem
	problem  = minimize(sumsquares(B - (A*X)), [X >= 0])
	
	# solve problem
	if solver == :SCS
		solve!(problem,SCSSolver(kwargs))
	elseif solver == :ECOS
		solve!(problem,ECOSSolver(kwargs))
	else
		error("solver symbol not recognized")
	end

	# check solution
	if problem.status != :Optimal
		warn("Solving for H problem status is :",problem.status)
	end

	# return solution
	return X.value
end
