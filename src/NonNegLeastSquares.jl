isdefined(Base, :__precompile__) && __precompile__()

module NonNegLeastSquares

export nonneg_lsq

## Algorithms
include("nnls.jl")
include("fnnls.jl")
include("convex.jl")
include("pivot.jl")
include("pivot_comb.jl")
include("pivot_cache.jl")

## Common interface to algorithms
include("interface.jl")

## Helper functions
include("cssls.jl") # combinatorial subspace least squares (CSSLS)

end