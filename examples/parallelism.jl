using NonNegLeastSquares
using OptimizationBenchmarks
using Plots; pyplot()

# parameters
n_replicates = 4 # trials per 
max_procs = 8

# get synthetic data for testing
A,X,B = nnls(;m=100,n=100,k=1000)
nonneg_lsq(A,B,alg=:fnnls) # burn in

# remove all worker processes
[ p != 1 && rmprocs(p) for p in procs()]

# run fits
t,p = [],[]
for np = 1:max_procs
    info(string(nprocs()," process"))
    for r = 1:n_replicates
        tic()
        @time nonneg_lsq(A,B,alg=:fnnls)
        if r >= 2
            push!(t,toc())
            push!(p,np)
        end
    end

    # Add a processor and go to next iteration
    addprocs(1)
    @everywhere using NonNegLeastSquares
end

plot(
    x=(p-1),y=t,
    line=(:none),
    marker=(:o),
    legend=(:none),
    title=("FNNLS runtime"),
    xlabel=("Number of worker processes"),
    ylabel=("Runtime (s)")
)
