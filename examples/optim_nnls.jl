## Comparison of Optim nnls and this package

using NonNegLeastSquares
import Optim

r = Optim.nnls(randn(5,5),randn(5))
b = nonneg_lsq(randn(5,5),randn(5))


A = randn(2000,2000)
b = randn(2000)

@time r = Optim.nnls(A,b)
@time b = nonneg_lsq(A,b) # roughly 2x faster
