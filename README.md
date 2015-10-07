# NonNegLeastSquares.jl
Some nonnegative least squares solvers in Julia

### Basic Usage:

The command `x = nonneg_lsq(A,b)` solves the optimization problem:

Minimize `||A*x - b||` subject to `xᵢ >= 0`, where `A` (a matrix) and `b` (a vector) are parameters and `x` (a vector) is the variable to be optimized.

### Currently Implemented Algorithms:

The code defaults to the "Fast NNLS" algorithm. To specify a different algorithm, use the keyword argument `alg`. Currently implemented algorithms are:

```julia
nonneg_lsq(A,b;alg=:nnls)  # NNLS
nonneg_lsq(A,b;alg=:fnnls) # Fast NNLS
nonneg_lsq(A,b;alg=:convex,solver=:SCS) # using Convex.jl with SCSSolver
nonneg_lsq(A,b;alg=:convex,solver=:SCS,verbose=false) # stops SCS from printing
nonneg_lsq(A,b;alg=:convex,solver=:ECOS) # using Convex.jl with ECOSSolver
```

Default behaviors:

```julia
nonneg_lsq(A,b) # Fast NNLS
nonneg_lsq(A,b;alg=:convex) # uses SCSSolver
```

***References***
* **NNLS**:
     * Lawson, C.L. and R.J. Hanson, Solving Least-Squares Problems, Prentice-Hall, Chapter 23, p. 161, 1974.
* **Fast NNLS**:
     * Bro R, De Jong S. [A fast non-negativitity-constrained least squares algorithm](https://dx.doi.org/10.1002%2F%28SICI%291099-128X%28199709%2F10%2911%3A5%3C393%3A%3AAID-CEM483%3E3.0.CO%3B2-L). Journal of Chemometrics. 11, 393–401 (1997)
* [**Convex.jl**](https://github.com/JuliaOpt/Convex.jl)
     * Udell et al. [Convex Optimization in Julia](https://web.stanford.edu/~boyd/papers/pdf/convexjl.pdf). SC14 Workshop on High Performance Technical Computing in Dynamic Languages. (2014)

### Installation:

```julia
Pkg.clone("https://github.com/ahwillia/NonNegLeastSquares.jl.git")

Pkg.test("NonNegLeastSquares")
```

### Example:

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

**NNLS** vs. **Fast NNLS**

```julia
@time nonneg_lsq(randn(200,200),randn(200),alg=:nnls)
#     4.752788 seconds (27.38 k allocations: 342.046 MB, 0.57% gc time)
```
```julia
@time nonneg_lsq(randn(200,200),randn(200),alg=:fnnls)
#     0.151799 seconds (23.58 k allocations: 13.199 MB, 1.11% gc time)
```

**Fast NNLS** vs. **Convex.jl**

```julia
@time nonneg_lsq(randn(1000,1000),randn(1000),alg=:fnnls)
#     5.414385 seconds (46.20 k allocations: 1.013 GB, 2.42% gc time)
```
```julia
@time nonneg_lsq(randn(1000,1000),randn(1000),alg=:convex,solver=:SCS)
#     18.688281 seconds (25.85 k allocations: 325.605 MB, 0.51% gc time)
```
```julia
@time nonneg_lsq(randn(1000,1000),randn(1000),alg=:convex,solver=:ECOS)
#     32.114776 seconds (23.72 k allocations: 299.724 MB, 0.22% gc time)
```

### Algorithims That Need Implementing:

* ftp://net9.cs.utexas.edu/pub/techreports/tr06-54.pdf
