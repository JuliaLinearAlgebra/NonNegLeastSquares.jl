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
	b::Vector{Float64};
	solver::Symbol = :SCS,
	kwargs...)
	
	# dimensions
	m,n = size(A)

	# solve problem
	x = Variable(n)
	problem  = minimize(sumsquares(b - (A*x)), [x >= 0])

	if solver == :SCS
		solve!(problem,SCSSolver(kwargs))
	elseif solver == :ECOS
		solve!(problem,ECOSSolver(kwargs))
	else
		error("solver symbol not recognized")
	end

	if problem.status != :Optimal
		warn("Solving for H problem status is :",problem.status)
	end

	# return solution
	return x.value
end