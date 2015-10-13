using Convex
using SCS
using ECOS


"""
x = convex(A, b, solver; kwargs...)

Solves the nonnegative least-squares problem using Convex.jl and the solver
specified by "solver". 
"""
function convex(
	A::Matrix{Float64},
	B::Matrix{Float64};
	variant::Symbol = :SCS,
	kwargs...)

	# Primal optimization variables
	X = Variable(size(A,2),size(B,2))
	
	# declare problem
	problem  = minimize(sumsquares(B - (A*X)), [X >= 0])
	
	# solve problem
	if variant == :SCS
		solve!(problem,SCSSolver(kwargs))
	elseif variant == :ECOS
		solve!(problem,ECOSSolver(kwargs))
	else
		error("Specified solver :",variant," not recognized")
	end

	# check solution
	if problem.status != :Optimal
		warn("Solving for H problem status is :",problem.status)
	end

	# return solution
	return X.value
end
