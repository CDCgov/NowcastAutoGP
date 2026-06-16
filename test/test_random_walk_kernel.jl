@testitem "RandomWalk eval_cov (scalar form)" begin
    using AutoGP

    node = RandomWalk(0.0, 2.0)
    # k(t, t') = amplitude * (min(t, t') - origin)
    @test AutoGP.GP.eval_cov(node, 0.3, 0.5) == 2.0 * 0.3
    @test AutoGP.GP.eval_cov(node, 0.5, 0.3) == 2.0 * 0.3   # symmetric
    @test AutoGP.GP.eval_cov(node, 0.4, 0.4) == 2.0 * 0.4   # diagonal

    # a non-zero origin shifts the covariance by amplitude * origin
    shifted = RandomWalk(-1.0, 1.0)
    @test AutoGP.GP.eval_cov(shifted, 0.3, 0.5) == (0.3 - (-1.0))

    # default amplitude is 1
    @test RandomWalk(0.0).amplitude == 1
end

@testitem "RandomWalk eval_cov (matrix form) matches scalar and is PSD" begin
    using AutoGP
    using Random: Random

    node = RandomWalk(0.0, 1.5)
    ts = collect(0.0:0.25:1.0)
    C = AutoGP.GP.eval_cov(node, ts)

    @test size(C) == (length(ts), length(ts))
    # matrix entries equal the pairwise scalar evaluation
    for i in eachindex(ts), j in eachindex(ts)
        @test C[i, j] == AutoGP.GP.eval_cov(node, ts[i], ts[j])
    end
    @test C == C'                                   # symmetric

    # positive semi-definite (origin = 0 <= min(ts)): vᵀ C v >= 0 for all v.
    # Checked without LinearAlgebra via the quadratic form on many random vectors.
    rng = Random.MersenneTwister(1234)
    quad(v) = sum(v[i] * C[i, j] * v[j] for i in eachindex(v), j in eachindex(v))
    for _ in 1:1000
        @test quad(randn(rng, length(ts))) >= -1.0e-10
    end
end

@testitem "RandomWalk reparameterize matches input warping" begin
    using AutoGP

    node = RandomWalk(-0.5, 2.0)
    t = AutoGP.GP.LinearTransform(3.0, -1.0)   # f(x) = 3x - 1, slope > 0
    warped = AutoGP.GP.reparameterize(node, t)

    # defining property: k(reparameterize(n, t), s, u) == k(n, f(s), f(u))
    for (s, u) in ((0.2, 0.7), (0.5, 0.5), (0.9, 0.1))
        fs = t.slope * s + t.intercept
        fu = t.slope * u + t.intercept
        @test AutoGP.GP.eval_cov(warped, s, u) ≈ AutoGP.GP.eval_cov(node, fs, fu)
    end

    # closed-form parameters (origin like Linear; amplitude is degree-1 in slope)
    @test warped.origin ≈ (node.origin - t.intercept) / t.slope
    @test warped.amplitude ≈ t.slope * node.amplitude
end

@testitem "RandomWalk rescale matches output scaling" begin
    using AutoGP

    node = RandomWalk(-0.5, 2.0)
    t = AutoGP.GP.LinearTransform(4.0, 7.0)   # output warp Y = 4X + 7
    scaled = AutoGP.GP.rescale(node, t)

    # output scaling multiplies the variance by slope^2; origin is unchanged
    @test scaled.origin == node.origin
    @test scaled.amplitude ≈ t.slope^2 * node.amplitude
    for (s, u) in ((0.2, 0.7), (0.5, 0.5))
        @test AutoGP.GP.eval_cov(scaled, s, u) ≈ t.slope^2 * AutoGP.GP.eval_cov(node, s, u)
    end
end

@testitem "RandomWalk pretty printing" begin
    using AutoGP

    @test AutoGP.GP.pretty(RandomWalk(0.0, 2.0)) == "RW(0.00; 2.00)"
    @test AutoGP.GP.pretty(RandomWalk(-1.5, 0.5)) == "RW(-1.50; 0.50)"
end
