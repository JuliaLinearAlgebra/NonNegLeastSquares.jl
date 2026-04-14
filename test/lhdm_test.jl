import NonNegLeastSquares.LHDM

const test_allocs = false

"""
Measure memory allocation within a function to avoid issues
with global variables.
"""
macro wrappedallocs(expr)
    argnames = [gensym() for a in expr.args]
    quote
        function g($(argnames...))
            @allocated $(Expr(expr.head, argnames...))
        end
        $(Expr(:call, :g, [esc(a) for a in expr.args]...))
    end
end


@testset "Float32 and BigFloat" begin
    Random.seed!(5)
    for i in 1:100
        m = rand(1:10)
        n = rand(1:10)
        A = randn(m, n)
        b = randn(m)
        x1 = @inferred LHDM.lhdm(A, b)
        x2 = @inferred LHDM.lhdm(Float32.(A), Float32.(b))
        x3 = @inferred LHDM.lhdm(BigFloat.(A), BigFloat.(b))
        @test x1 ≈ x2
        @test x1 ≈ x3
    end
end

if test_allocs
    @testset "lhdm allocations" begin
        Random.seed!(101)
        for i in 1:50
            m = rand(20:100)
            n = rand(20:100)
            A = randn(m, n)
            b = randn(m)
            work = @inferred LHDM.LHDMWorkspace(A, b)
            @test @wrappedallocs(LHDM.lhdm!(work)) == 0
        end
    end
end

@testset "lhdm workspace reuse" begin
    Random.seed!(200)
    m = 10
    n = 20
    work = @inferred LHDM.LHDMWorkspace(m, n)
    LHDM.lhdm!(work, randn(m, n), randn(m))
    for i in 1:100
        A = randn(m, n)
        b = randn(m)
        if test_allocs
            @test @wrappedallocs(LHDM.lhdm!(work, A, b)) == 0
        else
            LHDM.lhdm!(work, A, b)
        end
        @test work.x ≈ LHDM.lhdm(A, b)
    end

    m = 20
    n = 10
    for i in 1:100
        A = randn(m, n)
        b = randn(m)
        LHDM.lhdm!(work, A, b)
        @test work.x ≈ LHDM.lhdm(A, b)
    end
end

if test_allocs
    @testset "non-Int Integer workspace" begin
        m = 10
        n = 20
        A = randn(m, n)
        b = randn(m)
        work = @inferred LHDM.LHDMWorkspace(A, b, Int32)
        # Compile
        LHDM.lhdm!(work)

        A = randn(m, n)
        b = randn(m)
        work = @inferred LHDM.LHDMWorkspace(A, b, Int32)
        @test @wrappedallocs(LHDM.lhdm!(work)) == 0
    end
end

@testset "LHDM guaranteed positive exact solution and use_parallel" begin
    Random.seed!(101)
    m = 10
    n = 20
    for _ in 1:100
        A = randn(m, n)
        x = rand(n)
        b = A * x
        x_sparse = LHDM.lhdm(A, b)
        @test A * x_sparse ≈ b
    end
    k = 5
    for _ in 1:100
        A = randn(m, n)
        x = rand(n, k)
        b = A * x
        x_sparse = @inferred LHDM.lhdm(A, b, use_parallel=true)
        @test A * x_sparse ≈ b
    end
end