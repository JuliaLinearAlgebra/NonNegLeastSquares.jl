"""
    cssls!(AtA,AtB,X,P)

Solve combinatorial subspace least squares (CSSLS) problem. Find `X[P]` that
minimizes `||A*X[P] - B[P]||` in the least squares sense, where `P` is a
1-bit matrix encoding the combinatorial subspace. Input `X` is overwritten
with the solution.

In the context of nonnegative least squares, each column of `X` has a passive
set which is encoded by "true" entries in `P`,
and an active set which is encoded by "false" entries in `P`.

Inputs:
* `AtA = A' * A = (n x n)` matrix
* `AtB = A' * B = (n x k)`
* `P` = 1-bit matrix of Bools = (? x ?) matrix

References:
	J. Kim and H. Park, Fast nonnegative matrix factorization: An
	active-set-like method and comparisons, SIAM J. Sci. Comput., 33 (2011),
	pp. 3261–3281.

	M. H. Van Benthem and M. R. Keenan, Fast algorithm for the solution of
	large-scale non-negativity-constrained least squares problems. J.
	Chemometrics 2004; 18: 441-450
"""
function cssls!(
	AtA::AbstractMatrix, # (n x n) matrix
	AtB::AbstractMatrix, # (n x p) matrix
	X::AbstractMatrix,   # (n x p) matrix
	P::AbstractArray{Bool}
	)

	n,p = size(AtB)

	# Find unique columns in P
	U = unique(P,dims=2)
	num_unique = size(U,2)

	# Find indices associates with unique columns in P
	E = (Array{Int})[]
	rp = collect(1:p)
	for i = 1:num_unique
		# array of indices where P[:,e] == U[:,i]
		e = (Int)[]

		# find columns of P that match U[:,i]
		j = 1
		while j <= length(rp)
			# rp contains columns of P that haven't been assigned
			t = rp[j]

			# if column t matches U[:,i] store it, and delete it from rp (don't
			# revisit preiously assigned columns)
			if all(P[:,t] .== U[:,i])
				push!(e,t)      # store column t
				deleteat!(rp,j) # delete t from rp (stored at position j)
			else
				# column t didn't match, move to next column
				j += 1
			end
		end

		# store the set of columns matching U[:,i]
		push!(E,e)
	end

	# Update X, solve each unique combinatorial subspace
	for i = 1:num_unique
		rr = U[:,i] # row mask (the passive set)

		# If the passive set is empty, these columns are already feasible.
		# So don't waste time updating them.
		if any(rr)
			cc = E[i]   # column mask (columns of P matching U[:,i])
			X[rr,cc] = pinv(AtA[rr,rr])*AtB[rr,cc]
		end
	end
end
