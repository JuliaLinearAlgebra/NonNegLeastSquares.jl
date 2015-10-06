# NonNegLeastSquares.jl
Some nonnegative least squares solvers in Julia

### Basic Usage:

The command `x = nonneg_lsq(A,b)` solves the optimization problem:

Minimize `||A*x - b||` subject to `xᵢ >= 0`, where `A` (a matrix) and `b` (a vector) are parameters and `x` (a vector) is the variable to be optimized.

### Currently Implemented Algorithms:

Specify the algorithm with the keyword argument `alg`. Currently implemented algorithms are:

* **NNLS**: Lawson, C.L. and R.J. Hanson, Solving Least-Squares Problems, Prentice-Hall, Chapter 23, p. 161, 1974.

```julia
nonneg_lsq(A,b;alg=:nnls)
```

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

### Algorithims That Need Implementing:

* **Fast NNLS**: [Bro & De Jong (1997)](https://dx.doi.org/10.1002%2F%28SICI%291099-128X%28199709%2F10%2911%3A5%3C393%3A%3AAID-CEM483%3E3.0.CO%3B2-L)
