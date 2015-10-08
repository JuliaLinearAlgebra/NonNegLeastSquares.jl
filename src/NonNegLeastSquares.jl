module NonNegLeastSquares

export nonneg_lsq

## Algorithms
include("nnls.jl")
include("fnnls.jl")
include("convex.jl")
include("pivot.jl")

## Common interface to algorithms
include("interface.jl")

end