@testitem "IntegratedBrownianMotion eval_cov (scalar form)" begin
    using AutoGP

    node = IntegratedBrownianMotion(0.0, 2.0)
    # For origin = 0 and s <= t the covariance is amplitude * (s^2 t / 2 - s^3 / 6).
    s, t = 0.3, 0.5
    @test AutoGP.GP.eval_cov(node, s, t) ≈ 2.0 * (s^2 * t / 2 - s^3 / 6)
    @test AutoGP.GP.eval_cov(node, t, s) ≈ AutoGP.GP.eval_cov(node, s, t)   # symmetric
    @test AutoGP.GP.eval_cov(node, 0.0, 0.4) == 0.0                         # zero variance at origin

    # a non-zero origin shifts time: only (t - origin) enters the kernel
    shifted = IntegratedBrownianMotion(-1.0, 1.0)
    a, b = 0.3 - (-1.0), 0.5 - (-1.0)
    @test AutoGP.GP.eval_cov(shifted, 0.3, 0.5) ≈ a^2 * (3b - a) / 6

    # default amplitude is 1
    @test IntegratedBrownianMotion(0.0).amplitude == 1
end

@testitem "IntegratedBrownianMotion eval_cov (matrix form) matches scalar and is PSD" begin
    using AutoGP
    using Random: Random

    node = IntegratedBrownianMotion(0.0, 1.5)
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

@testitem "IntegratedBrownianMotion reparameterize matches input warping" begin
    using AutoGP

    node = IntegratedBrownianMotion(-0.5, 2.0)
    t = AutoGP.GP.LinearTransform(3.0, -1.0)   # f(x) = 3x - 1, slope > 0
    warped = AutoGP.GP.reparameterize(node, t)

    # defining property: k(reparameterize(n, t), s, u) == k(n, f(s), f(u))
    for (s, u) in ((0.2, 0.7), (0.5, 0.5), (0.9, 0.1))
        fs = t.slope * s + t.intercept
        fu = t.slope * u + t.intercept
        @test AutoGP.GP.eval_cov(warped, s, u) ≈ AutoGP.GP.eval_cov(node, fs, fu)
    end

    # closed-form parameters (origin like RandomWalk; amplitude is degree-3 in slope)
    @test warped.origin ≈ (node.origin - t.intercept) / t.slope
    @test warped.amplitude ≈ t.slope^3 * node.amplitude
end

@testitem "IntegratedBrownianMotion rescale matches output scaling" begin
    using AutoGP

    node = IntegratedBrownianMotion(-0.5, 2.0)
    t = AutoGP.GP.LinearTransform(4.0, 7.0)   # output warp Y = 4X + 7
    scaled = AutoGP.GP.rescale(node, t)

    # output scaling multiplies the variance by slope^2; origin is unchanged
    @test scaled.origin == node.origin
    @test scaled.amplitude ≈ t.slope^2 * node.amplitude
    for (s, u) in ((0.2, 0.7), (0.5, 0.5))
        @test AutoGP.GP.eval_cov(scaled, s, u) ≈ t.slope^2 * AutoGP.GP.eval_cov(node, s, u)
    end
end

@testitem "IntegratedBrownianMotion pretty printing" begin
    using AutoGP

    @test AutoGP.GP.pretty(IntegratedBrownianMotion(0.0, 2.0)) == "IBM(0.00; 2.00)"
    @test AutoGP.GP.pretty(IntegratedBrownianMotion(-1.5, 0.5)) == "IBM(-1.50; 0.50)"
end
