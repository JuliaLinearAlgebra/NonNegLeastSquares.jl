module NnlsTest

using Base.Test
using PyCall
const pyopt = pyimport_conda("scipy.optimize", "scipy")
using NonNegLeastSquares.NNLS

macro wrappedallocs(expr)
    argnames = [gensym() for a in expr.args]
    quote
        function g($(argnames...))
            @allocated $(Expr(expr.head, argnames...))
        end
        $(Expr(:call, :g, [esc(a) for a in expr.args]...))
    end
end

@testset "apply_householder!" begin
    srand(2)
    for i in 1:10
        u = randn(rand(3:10))
        c = randn(length(u))
        
        u1 = copy(u)
        c1 = copy(c)
        up1 = NNLS.construct_householder!(u1, 0.0)
        NNLS.apply_householder!(u1, up1, c1)

        u2 = copy(u)
        c2 = copy(c)
        @test @wrappedallocs(NNLS.construct_householder!(u2, 0.0)) == 0
        up2 = up1
        @test @wrappedallocs(NNLS.apply_householder!(u2, up2, c2)) == 0
    end
end

@testset "orthogonal_rotmat" begin
    srand(3)
    for i in 1:1000
        a = randn()
        b = randn()
        c, s, sig = NNLS.orthogonal_rotmat(a, b)
        @test [c s; -s c] * [a, b] ≈ [sig, 0]
        @test @wrappedallocs(NNLS.orthogonal_rotmat(a, b)) == 0
    end
end

@testset "nnls allocations" begin
    srand(101)
    for i in 1:50
        m = rand(20:100)
        n = rand(20:100)
        A = randn(m, n)
        b = randn(m)
        work = NNLSWorkspace(A, b)
        @test @wrappedallocs(nnls!(work)) == 0
    end
end

@testset "nnls workspace reuse" begin
    srand(200)
    m = 10
    n = 20
    work = NNLSWorkspace(m, n)
    nnls!(work, randn(m, n), randn(m))
    for i in 1:100
        A = randn(m, n)
        b = randn(m)
        @test @wrappedallocs(nnls!(work, A, b)) == 0
        @test work.x == pyopt[:nnls](A, b)[1]
    end

    m = 20
    n = 10
    for i in 1:100
        A = randn(m, n)
        b = randn(m)
        nnls!(work, A, b)
        @test work.x == pyopt[:nnls](A, b)[1]
    end
end

@testset "non-Int Integer workspace" begin
    m = 10
    n = 20
    A = randn(m, n)
    b = randn(m)
    work = NNLSWorkspace(A, b, Int32)
    # Compile
    nnls!(work)

    A = randn(m, n)
    b = randn(m)
    work = NNLSWorkspace(A, b, Int32)
    @test @wrappedallocs(nnls!(work)) <= 0
end

@testset "nnls vs scipy" begin
    srand(5)
    for i in 1:5000
        m = rand(1:60)
        n = rand(1:60)
        A = randn(m, n)
        b = randn(m)
        x1 = nnls(A, b)
        x2, residual2 = pyopt[:nnls](A, b)
        @test x1 == x2
    end
end

end
