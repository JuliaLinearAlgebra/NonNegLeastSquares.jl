module LHDM

using LinearAlgebra

export lhdm,
       lhdm!,
       LHDMWorkspace,
       load!

include("nnls.jl")
import .NNLS: construct_householder!, apply_householder!,
              orthogonal_rotmat, solve_triangular_system! 

mutable struct LHDMWorkspace{T, I <: Integer}
    QA::Matrix{T}
    Qb::Vector{T}
    u::Vector{T}
    x::Vector{T}
    w::Vector{T}
    zz::Vector{T}
    idx::Vector{I}
    I::Vector{I}
    rnorm::T
    mode::I
    nsetp::I
end

function LHDMWorkspace{T,I}(m, n) where {T, I<:Integer}
    LHDMWorkspace{T,I}(Matrix{T}(undef, m, n), # A
                       Vector{T}(undef,m),    # b
                       Vector{T}(undef,n),    # n
                       Vector{T}(undef,n),    # x
                       Vector{T}(undef,n),    # w
                       Vector{T}(undef,m),    # zz
                       Vector{I}(undef,n),    # idx
                       Vector{I}(undef,m),    # I
                       zero(T), # rnorm
                       zero(I), # mode
                       zero(I)  # nsetp
       )
end
LHDMWorkspace{T}(m, n) where T = LHDMWorkspace{T, Int}(m, n)

function Base.resize!(work::LHDMWorkspace{T}, m::Integer, n::Integer) where {T}
    work.QA = Matrix{T}(undef,m, n)
    resize!(work.Qb, m)
    resize!(work.u, n)
    resize!(work.x, n)
    resize!(work.w, n)
    resize!(work.zz, m)
    resize!(work.idx, n)
    resize!(work.I, m)
end

function load!(work::LHDMWorkspace{T}, A::AbstractMatrix{T}, b::AbstractVector{T}) where {T}
    m, n = size(A)
    @assert size(b) == (m,)
    if size(work.QA, 1) != m || size(work.QA, 2) != n
        resize!(work, m, n)
    end
    work.QA .= A
    work.Qb .= b
    work
end

# using this is not recommended, just use LHDMWorkspace{T, I}(m, n) directly.
LHDMWorkspace(m::Integer, n::Integer,
                    eltype::Type{T}=Float64,
                    indextype::Type{I}=Int) where {T,I} = LHDMWorkspace{T, I}(m, n)

function LHDMWorkspace(A::Matrix{T}, b::Vector{T}, indextype::Type{I}=Int) where {T,I}
    m, n = size(A)
    @assert size(b) == (m,)
    work = LHDMWorkspace{T, I}(m, n)
    load!(work, A, b)
    work
end

@noinline function checkargs(work::LHDMWorkspace)
    m, n = size(work.QA)
    @assert size(work.Qb) == (m,)
    @assert size(work.u) == (n,)
    @assert size(work.x) == (n,)
    @assert size(work.w) == (n,)
    @assert size(work.zz) == (m,)
    @assert size(work.idx) == (n,)
end


"""
Algorithm LHDM: LAWSON HANSON DEVIATION MAXIMIZATION FOR NONNEGATIVE LEAST SQUARES

This code is a modified version of nnls.jl here which 
uses deviation maximization to add multiple indices
at a time. The algorithm is in the following paper:
https://doi.org/10.1002/nla.2490
"""
function lhdm!(work::LHDMWorkspace{T, TI},
               max_iter::Integer=(3 * size(work.QA, 2));
               kmax::Int = 32,
               thres_w::Real = 0.6,
               thres_nrm::Real = 0.15,
               thres_cos::Real = 0.9) where {T, TI}
    checkargs(work)

    @assert kmax >= 1
    @assert 0.0 ≤ thres_w ≤ 1.0
    @assert 0.0 ≤ thres_nrm ≤ 1.0
    @assert 0.0 ≤ thres_cos ≤ 1.0

    A = work.QA
    b = work.Qb
    u = work.u
    x = work.x
    w = work.w
    zz = work.zz
    idx = work.idx
    I = view(work.I, 1:min(kmax,length(work.I)))
    work.mode = 1

    m = convert(TI, size(A, 1))
    n = convert(TI, size(A, 2))
    minmn = min(m,n)

    iter = 0
    x .= 0
    idx .= 1:n
    I .= 0

    iz2 = n
    iz1 = one(TI)
    iz = zero(TI)
    j = zero(TI)
    jj = zero(TI)
    nsetp = zero(TI)
    up = zero(T)

    terminated = false

    # ******  MAIN LOOP BEGINS HERE  ******
    while true
        # QUIT IF ALL COEFFICIENTS ARE ALREADY IN THE SOLUTION.
        # OR IF M COLS OF A HAVE BEEN TRIANGULARIZED.
        if (iz1 > iz2 || nsetp >= m)
            terminated = true
            break
        end

        # COMPUTE COMPONENTS OF THE DUAL (NEGATIVE GRADIENT) VECTOR W().
        @inbounds for i in iz1:iz2
            idxi = idx[i]
            sm = zero(T)
            for l in (nsetp + 1):m
                sm += A[l, idxi] * b[l]
            end
            w[idxi] = sm
        end

        # COMPUTE U AS COLNORMS OF A
        u .= -Inf
        umax = -1.0
        @inbounds for jz in iz1:iz2
            jj = idx[jz]
            u[jj] = zero(T)
            for i in 1:m
                u[jj] += A[i,jj] ^ 2
            end
            u[jj] = sqrt(u[jj])
            if u[jj] > umax
                umax = u[jj]
            end
        end

        # FIND LARGEST POSITIVE W(J).
        wmax, izidx = findmax(iz -> w[idx[iz]], iz1:iz2)
        izmax = (iz1:iz2)[izidx]

        # IF WMAX .LE. 0. GO TO TERMINATION.
        # THIS INDICATES SATISFACTION OF THE KUHN-TUCKER CONDITIONS.
        if wmax <= 0
            terminated = true
            break
        end
        # FIND OTHER DEVIATION MAXIMIZATION INDS
        dm_ct = 2
        I .= 0
        I[1] = izmax
        for jz in iz1:iz2
            if dm_ct > min(kmax, minmn - nsetp)
                break
            end
            if jz == izmax
                continue
            end
            idxj = idx[jz]
            if w[idxj] > thres_w * wmax && u[idxj] > thres_nrm * umax
                I[dm_ct] = jz
                dm_ct += 1
            end
        end
        I_len = dm_ct - 1
        # RESTRICT DEVIATION MAXIMIZATION INDICES BY COS MAT
        iI = 1
        while iI < length(I)
            iI += 1
            if I[iI] == 0
                break
            end
            idxiI = idx[I[iI]]
            add = true
            for jI in 1:iI-1
                idxjI = idx[I[jI]]
                theta_ij = zero(T)
                # Compute dot product
                for i in 1:m
                    theta_ij += A[i, idxiI] * A[i, idxjI]
                end
                # Divide by norms
                theta_ij = theta_ij / (u[idxiI] * u[idxjI])
                if abs(theta_ij) > thres_cos
                    add = false
                    break
                end
            end
            if !add
                I[iI:I_len-1] .= @view I[iI+1:I_len]
                I[I_len] = 0
                iI -= 1
                I_len -= 1
            end
        end

        # THE MULTIPLE INDICES IN I HAVE BEEN SELECTED TO BE MOVED FROM
        # SET Z TO SET P.    UPDATE B,  UPDATE INDICES,  APPLY HOUSEHOLDER
        # TRANSFORMATIONS TO COLS IN NEW SET Z,  ZERO SUBDIAGONAL ELTS IN
        # COL J,  SET W(J)=0.
        zz .= b
        
        # ADD EACH INDEX FROM I TO P
        for iI in 1:I_len
            if I[iI] == 0
                break
            end
            iz = I[iI]
            j = idx[iz]
            ## FORM HOUSEHOLDER
            up = construct_householder!(
                view(A, nsetp+1:m, j), up)
            ## APPLY TO Qᵀb
            apply_householder!(
                view(A, nsetp+1:m, j),
                up,
                view(zz, nsetp+1:m))
            ## SHIFT AROUND INDICES
            idx[iz] = idx[iz1]
            idx[iz1] = j
            for iJ in eachindex(I)
                if I[iJ] == iz1
                    I[iJ] = iz
                    break
                end
            end
            I[iI] = iz1
            iz1 += one(TI)
            nsetp += one(TI)
            ## APPLY TO A
            if iz1 <= iz2
                for jz in iz1:iz2
                    jj = idx[jz]
                    apply_householder!(
                        view(A, nsetp:m, j),
                        up,
                        view(A, nsetp:m, jj))
                end
            end
            if nsetp != m
                for l in (nsetp + 1):m
                    A[l, j] = 0
                end
            end
            w[j] = 0
        end
        
        b .= zz

        # SOLVE THE TRIANGULAR SYSTEM.
        # STORE THE SOLUTION TEMPORARILY IN ZZ().
        jj = solve_triangular_system!(zz, A, idx, nsetp, jj)
        
        # ******  SECONDARY LOOP BEGINS HERE ******
        #
        # ITERATION COUNTER.
        while true
            iter += 1
            if iter > max_iter
                work.mode = 3
                terminated = true
                break
            end

            # CHECK IF ANY NEWLY ADDED INDICES NEGATIVE, IF SO 
            # REMOVE LAST ELEMENT ADDED
            any_I_neg = false
            if I_len > 1
                for i in (nsetp-I_len+1):nsetp
                    if zz[i] < 0
                        any_I_neg = true
                    end
                end
            end
            if any_I_neg
                x[idx[nsetp]] = 0
                I[I_len] = 0
                I_len -= one(TI)
                nsetp -= one(TI)
                iz1 -= one(TI)
                # COPY B INTO ZZ, RESOLVE SYSTEM, STORE IN ZZ
                zz .= b
                jj = solve_triangular_system!(zz, A, idx, nsetp, jj)
                continue
            end

            # SEE IF ALL NEW CONSTRAINED COEFFS ARE FEASIBLE.
            # IF NOT COMPUTE ALPHA.
            alpha = convert(T, 2)
            for ip in one(TI):nsetp
                l = idx[ip]
                if zz[ip] <= 0
                    t = -x[l] / (zz[ip] - x[l])
                    if alpha > t
                        alpha = t
                        jj = ip
                    end
                end
            end

            # IF ALL NEW CONSTRAINED COEFFS ARE FEASIBLE THEN ALPHA WILL
            # STILL = 2.    IF SO EXIT FROM SECONDARY LOOP TO MAIN LOOP.
            if alpha == 2
                break
            end

            # OTHERWISE USE ALPHA WHICH WILL BE BETWEEN 0 AND 1 TO
            # INTERPOLATE BETWEEN THE OLD X AND THE NEW ZZ.
            for ip in one(TI):nsetp
                l = idx[ip]
                x[l] = x[l] + alpha * (zz[ip] - x[l])
            end

            # MODIFY A AND B AND THE INDEX ARRAYS TO MOVE COEFFICIENT I
            # FROM SET P TO SET Z.
            i = idx[jj]
            kk = 1
            while true
                x[i] = 0

                if jj != nsetp
                    jj += one(TI)
                    for j in jj:nsetp
                        ii = idx[j]
                        idx[j - 1] = ii
                        cc, ss, sig = orthogonal_rotmat(A[j - 1, ii], A[j, ii])
                        A[j - 1, ii] = sig
                        A[j, ii] = 0
                        for l in one(TI):n
                            if l != ii
                                # Apply procedure G2 (CC,SS,A(J-1,L),A(J,L))
                                temp = A[j - 1, l]
                                A[j - 1, l] = cc * temp + ss * A[j, l]
                                A[j, l] = -ss * temp + cc * A[j, l]
                            end
                        end

                        # Apply procedure G2 (CC,SS,B(J-1),B(J))
                        temp = b[j - 1]
                        b[j - 1] = cc * temp + ss * b[j]
                        b[j] = -ss * temp + cc * b[j]
                    end
                end

                nsetp -= one(TI)
                iz1 -= one(TI)
                idx[iz1] = i

                # SEE IF THE REMAINING COEFFS IN SET P ARE FEASIBLE.  THEY SHOULD
                # BE BECAUSE OF THE WAY ALPHA WAS DETERMINED.
                # IF ANY ARE INFEASIBLE IT IS DUE TO ROUND-OFF ERROR.  ANY
                # THAT ARE NONPOSITIVE WILL BE SET TO ZERO
                # AND MOVED FROM SET P TO SET Z.
                allfeasible = true
                for jj in one(TI):nsetp
                    i = idx[jj]
                    if x[i] <= 0
                        allfeasible = false
                        kk = jj
                        break
                    end
                end
                if allfeasible
                    break
                end
            end


            # COPY B( ) INTO ZZ( ).  THEN SOLVE AGAIN AND LOOP BACK.
            zz .= b
            jj = solve_triangular_system!(zz, A, idx, nsetp, kk)
        end
        if terminated
            break
        end
        # ******  END OF SECONDARY LOOP  ******

        for i in 1:nsetp
            x[idx[i]] = zz[i]
        end
        # ALL NEW COEFFS ARE POSITIVE.  LOOP BACK TO BEGINNING.
    end

    # ******  END OF MAIN LOOP  ******
    # COME TO HERE FOR TERMINATION.
    # COMPUTE THE NORM OF THE FINAL RESIDUAL VECTOR.

    sm = zero(T)
    if nsetp < m
        for i in (nsetp + 1):m
            sm += b[i]^2
        end
    else
        w .= 0
    end
    work.rnorm = sqrt(sm)
    work.nsetp = nsetp
    return work.x
end

function lhdm!(work::LHDMWorkspace{T},
               A::AbstractMatrix{T},
               b::AbstractVector{T},
               args...; kwargs...) where {T}
    load!(work, A, b)
    lhdm!(work, args...; kwargs...)
    work.x
end


"""
    x = lhdm(A, b; ...)

Solves non-negative least-squares (NNLS) problem
by "Lawson-Hanson algorithm with deviation maximization" (LHDM)
as published by Dessole et al. in 2023:
https://doi.org/10.1002/nla.2490
Optional arguments:
* `max_iter=3*size(A,2)`: maximum number of iterations (counts inner loop iterations)
Optional keyword arguments:
* `use_parallel::Bool=true`: if `b` is a `AbstractMatrix`, parallelizes calls over columns of `b`
* `kmax=32`: maximum number of indices added at a time (k=1 for standard LH-NNLS)
* `thres_w=0.6`: threshold factor  on dual vector for addition (between 0 and 1)
* `thres_nrm=0.15`: threshold factor on norm of columns for addition (between 0 and 1)
* `thres_cos=0.9`: threshold factor on cosine matrix (orthogonality) for addition (between 0 and 1)
"""
function lhdm(A,
              b::AbstractVector{T},
              args...; kwargs...) where {T}
    work = LHDMWorkspace(A, b)
    lhdm!(work, args...; kwargs...)
    work.x
end

function lhdm(A,
              B::AbstractMatrix{T},
              args...;
              use_parallel::Bool = true,
              kwargs...) where {T}

    m, n = size(A)
    k = size(B, 2)

    X = Array{T}(undef,n, k)
    if k > 1 && use_parallel && Threads.nthreads() > 1
        chunksize = ceil(Int, k / Threads.nthreads())
        colstarts = 1:chunksize:k
        tasks = Vector{Task}(undef, length(colstarts))
        for (i, colstart) in enumerate(colstarts)
            tasks[i] = Threads.@spawn begin
                colend = min(colstart + chunksize - 1, k)
                work = LHDMWorkspace{T}(m, n)
                for col in colstart:colend
                    X[:,col] = lhdm!(work, A, @view(B[:,col]), args...; kwargs...)
                end
            end
        end
        foreach(fetch, tasks)
    else
        let work = LHDMWorkspace{T}(m, n)
            for i = 1:k
                X[:, i] = lhdm!(work, A, @view(B[:,i]), args...; kwargs...)
            end
        end
    end
    return X
end

end # module
