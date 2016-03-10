

# wrapper function for convienence
nnls(A,b;Gram=false) = nonneg_lsq(A,b;alg=:nnls,Gram=Gram)

# Solve A*x = b for x, subject to x >=0 
A = [ 0.53879488  0.65816267 
      0.12873446  0.98669198
      0.24555042  0.00598804
      0.80491791  0.32793762 ]

b = [0.888,  0.562,  0.255,  0.077]

# Test that nnls produces the same solution as scipy
x = [0.15512102, 0.69328985] # approx solution from scipy
@test norm(nnls(A,b)-x) < 1e-5
@test norm(nnls(A'*A,A'*b;Gram=true)-x) < 1e-5


## A second test case
A2 = [ -0.24  -0.82   1.35   0.36   0.35
       -0.53  -0.20  -0.76   0.98  -0.54
        0.22   1.25  -1.60  -1.37  -1.94
       -0.51  -0.56  -0.08   0.96   0.46
        0.48  -2.25   0.38   0.06  -1.29 ]
b2 = [-1.6,  0.19,  0.17,  0.31, -1.27]
x2 = [2.2010416, 1.19009924, 0.0, 1.55001345, 0.0]
@test norm(nnls(A2,b2)-x2) < 1e-5
@test norm(nnls(A2'*A2,A2'*b2;Gram=true)-x2) < 1e-5

## Test a bunch of random cases
@pyimport scipy.optimize as pyopt

for i = 1:10
	m,n = rand(1:10),rand(1:10)
	A3 = randn(m,n)
	b3 = randn(m)
	x3,resid = pyopt.nnls(A3,b3)
	@test norm(nnls(A3,b3)-x3) < 1e-5
	@test norm(nnls(A3'*A3,A3'*b3;Gram=true)-x3) < 1e-5
end
