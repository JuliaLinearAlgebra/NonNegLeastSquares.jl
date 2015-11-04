[![Build Status](https://travis-ci.org/ahwillia/NonNegLeastSquares.jl.svg)](https://travis-ci.org/ahwillia/NonNegLeastSquares.jl?branch=master)

# NonNegLeastSquares.jl
Some nonnegative least squares solvers in Julia

### Basic Usage:

The command `X = nonneg_lsq(A,B)` solves the optimization problem:

Minimize `||A*X - B||` subject to `Xᵢⱼ >= 0`; in this case, `||.||` denotes the Frobenius norm (equivalently, the Euclidean norm if `B` is a column vector). The arguments `A` and `B` are respectively (m x k) and (m x n) matrices, so `X` is a (k x n) matrix.

### Currently Implemented Algorithms:

The code defaults to the "Fast NNLS" algorithm. To specify a different algorithm, use the keyword argument `alg`. Currently implemented algorithms are:

```julia
nonneg_lsq(A,b;alg=:nnls)  # NNLS
nonneg_lsq(A,b;alg=:fnnls) # Fast NNLS
nonneg_lsq(A,b;alg=:pivot) # Pivot Method
nonneg_lsq(A,b;alg=:pivot,variant=:cache) # Pivot Method (cache pseudoinverse up front)
nonneg_lsq(A,b;alg=:pivot,variant=:comb) # Pivot Method with combinatorial least-squares
nonneg_lsq(A,b;alg=:admm) # Alternating Direction Method of Multipliers
```

Default behaviors:

```julia
nonneg_lsq(A,b) # pivot method
```

***References***
* **NNLS**:
     * Lawson, C.L. and R.J. Hanson, Solving Least-Squares Problems, Prentice-Hall, Chapter 23, p. 161, 1974.
* **Fast NNLS**:
     * Bro R, De Jong S. [A fast non-negativitity-constrained least squares algorithm](https://dx.doi.org/10.1002%2F%28SICI%291099-128X%28199709%2F10%2911%3A5%3C393%3A%3AAID-CEM483%3E3.0.CO%3B2-L). Journal of Chemometrics. 11, 393–401 (1997)
* **Pivot Method**:
     * Kim J, Park H. [Fast nonnegative matrix factorization: an active-set-like method and comparisons](http://www.cc.gatech.edu/~hpark/papers/SISC_082117RR_Kim_Park.pdf). SIAM Journal on Scientific Computing 33.6 (2011): 3261-3281.
* **ADMM**:
	 * S. Boyd, N. Parikh, E. Chu, B. Peleato, and J. Eckstein (2011). Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers. Foundations and Trends in Machine Learning.

Note that there are other ways of solving nonnegative least-squares problems in Julia. For example, see the [**Convex.jl**](https://github.com/JuliaOpt/Convex.jl) package; check out the `convex_nnls` function available in the `examples/` directory. Also check out [the nnls solver in **Optim.jl**](https://github.com/JuliaOpt/Optim.jl#nonnegative-least-squares). The active set methods implemented here appear to be faster in many cases.

### Installation:

```julia
Pkg.add("NonNegLeastSquares")
Pkg.test("NonNegLeastSquares")
```

### Simple Example:

```julia
using NonNegLeastSquares

A = [ -0.24  -0.82   1.35   0.36   0.35
      -0.53  -0.20  -0.76   0.98  -0.54
       0.22   1.25  -1.60  -1.37  -1.94
      -0.51  -0.56  -0.08   0.96   0.46
       0.48  -2.25   0.38   0.06  -1.29 ];

b = [-1.6,  0.19,  0.17,  0.31, -1.27];

x = nonneg_lsq(A,b)
```

Produces:

```
5-element Array{Float64,1}:
 2.20104
 1.1901 
 0.0    
 1.55001
 0.0  
```

### Speed Comparisons:

Run the `examples/performance_check.jl` script to compare the various algorithms that have been implemented on some synthetic data. Note that the variants `:cache` and `:comb` of the pivot method improve performance substantially when the inner dimension, `k`, is small. For example, when `m = 5000` and `n = 5000` and `k=3`:

```
Comparing pivot:none to pivot:comb with A = randn(5000,3) and B = randn(5000,5000)
-------------------------------------------------------------------------------------
PIVOT:none →   2.337322 seconds (1.09 M allocations: 4.098 GB, 22.74% gc time)
PIVOT:comb →   0.096450 seconds (586.76 k allocations: 23.569 MB, 3.01% gc time)
```

### Algorithims That Need Implementing:

Pull requests are more than welcome, whether it is improving existing algorithms, or implementing new ones.

* ftp://net9.cs.utexas.edu/pub/techreports/tr06-54.pdf
* Sra Suvrit Kim Dongmin and Inderjit S. Dhillon. A non-monotonic method for large-scale non-negative least squares. Optimization Methods and Software, 28(5):1012–1039, 2013.
