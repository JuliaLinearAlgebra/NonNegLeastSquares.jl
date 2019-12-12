isdefined(Base, :__precompile__) && __precompile__()

module NonNegLeastSquares

using Distributed
using LinearAlgebra
import SparseArrays

export nonneg_lsq

## Algorithms
include("nnls.jl")
using .NNLS: nnls
include("fnnls.jl")
include("pivot.jl")
include("pivot_comb.jl")
include("pivot_cache.jl")
include("admm.jl")

## Common interface to algorithms
include("interface.jl")

## Helper functions
include("cssls.jl") # combinatorial subspace least squares (CSSLS)

end
