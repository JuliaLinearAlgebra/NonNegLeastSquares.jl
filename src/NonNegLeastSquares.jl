module NonNegLeastSquares

export nonneg_lsq

## Algorithms
include("nnls.jl")
include("fnnls.jl")
include("convex.jl")
include("pivot_mrhs.jl")
include("pivot_srhs.jl")

## Common interface to algorithms
include("interface.jl")

## Helper functions
include("cssls.jl") # combinatorial subspace least squares (CSSLS)

end