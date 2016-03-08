isdefined(Base, :__precompile__) && __precompile__()

module NonNegLeastSquares

export nonneg_lsq

## Algorithms
include("nnls.jl")
include("nnls_gram.jl")
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
