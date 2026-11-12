@testsnippet CountObservationData begin
    using Dates
    using Random
    Random.seed!(87)  # For reproducible test data

    # Weekly counts with a seasonal rate that falls close to zero in the trough.
    count_dates = collect(Date(2023, 1, 7):Week(1):Date(2024, 3, 30))
    rate = exp.(1.0 .+ 1.4 .* sin.(2pi .* (1:length(count_dates)) ./ 52))
    counts = NowcastAutoGP._sample_counts(log.(rate))

    count_params = (n_particles = 2, n_mcmc = 10, n_hmc = 5)
    count_forecast_dates = count_dates[end] .+ Week.(1:4)

    count_data = create_count_data(count_dates, counts)
    count_model = make_and_fit_model(count_data; count_params...)
end

@testitem "CountData construction and validation" begin
    using Dates
    dates = [Date(2024, 1, 1), Date(2024, 1, 8), Date(2024, 1, 15)]

    data = CountData(dates, [0, 3, 12])
    @test data.ds == dates
    @test data.values == [0.0, 3.0, 12.0]
    @test data.values isa Vector{Float64}
    @test data.observation_model isa PoissonObservation

    # Convenience constructor accepts any iterables, and non-integer values.
    data = create_count_data(Date(2024, 1, 1):Week(1):Date(2024, 1, 15), (0.5, 3, 12))
    @test data.ds == dates
    @test data.values == [0.5, 3.0, 12.0]

    @test_throws ArgumentError CountData(dates, [1, 2])
    @test_throws ArgumentError CountData(dates, [1, -2, 3])
    @test_throws ArgumentError CountData(dates, [1.0, NaN, 3.0])
    @test_throws ArgumentError CountData(dates, [1.0, Inf, 3.0])
end

@testitem "Poisson pseudo-observations" begin
    using NowcastAutoGP: PoissonFixedSites, _site_mean, _site_var, _sites
    digamma, trigamma = NowcastAutoGP.digamma, NowcastAutoGP.trigamma

    # Moments of the Poisson likelihood as a function of the log-intensity.
    @test _site_mean(0) == digamma(0.5)
    @test _site_var(3) == trigamma(3.5)
    # For large counts these approach log(y) and 1 / y.
    @test isapprox(_site_mean(1000), log(1000); atol = 1.0e-6)
    @test isapprox(_site_var(1000), 1 / 1000; rtol = 1.0e-5)

    # Link constants centre and scale the pseudo-observations like AutoGP does for data.
    y = [0, 2, 5, 40, 7]
    d = PoissonFixedSites(y)
    z, v = _sites(d, y)
    @test isapprox(sum(z), 0.0; atol = 1.0e-12)
    @test maximum(z) - minimum(z) ≈ 1.0
    @test v ≈ _site_var.(y) ./ d.c1^2
    @test all(>(0), v)

    # A series that is flat within its count noise is scaled by that noise instead.
    d_flat = PoissonFixedSites(zeros(10))
    @test d_flat.c0 ≈ digamma(0.5)
    @test d_flat.c1 ≈ sqrt(trigamma(0.5))
    z_flat, v_flat = _sites(d_flat, zeros(10))
    @test all(z -> abs(z) < 1.0e-12, z_flat)
    @test all(isfinite, v_flat)
end

@testitem "PoissonFixedSites log density and gradient" begin
    using LinearAlgebra
    using Random
    using NowcastAutoGP: PoissonFixedSites, _sites
    Gen = NowcastAutoGP.Gen
    Random.seed!(1)

    n = 12
    ts = collect(range(0, 1, length = n))
    K = [0.7 * exp(-abs(a - b) / 0.3) for a in ts, b in ts]
    noise = 0.05
    y = Float64[0, 1, 0, 3, 6, 9, 14, 11, 5, 2, 0, 1]
    d = PoissonFixedSites(y)
    z, v = _sites(d, y)

    # The density is a Gaussian on the pseudo-observations with their own noise variances.
    expected = Gen.logpdf(Gen.mvnormal, z, zeros(n), K + noise * I + Diagonal(v))
    @test Gen.logpdf(d, y, K, noise) ≈ expected

    # Gradient against central finite differences.
    (dy, dK, dnoise) = Gen.logpdf_grad(d, y, K, noise)
    @test isnothing(dy)
    @test size(dK) == (n, n)
    h = 1.0e-6
    fd_noise = (Gen.logpdf(d, y, K, noise + h) - Gen.logpdf(d, y, K, noise - h)) / (2h)
    @test isapprox(dnoise, fd_noise; rtol = 1.0e-5)
    for _ in 1:5
        E = randn(n, n)
        E = (E + E') / 2
        fd_K = (Gen.logpdf(d, y, K + h * E, noise) - Gen.logpdf(d, y, K - h * E, noise)) /
            (2h)
        @test isapprox(sum(dK .* E), fd_K; rtol = 1.0e-5)
    end

    # No observations, as at the start of sequential Monte Carlo.
    @test Gen.logpdf(d, Float64[], zeros(0, 0), noise) == 0.0
    @test Gen.logpdf_grad(d, Float64[], zeros(0, 0), noise) == (nothing, zeros(0, 0), 0.0)
    @test Gen.random(d, zeros(0, 0), noise) == Float64[]

    # Invalid counts have zero density rather than throwing.
    @test Gen.logpdf(d, [1.0, -1.0], K[1:2, 1:2], noise) == -Inf

    # Simulation draws counts.
    draw = Gen.random(d, K, noise)
    @test draw isa Vector{Float64}
    @test length(draw) == n
    @test all(x -> x >= 0 && isinteger(x), draw)
    @test Gen.has_argument_grads(d) == (true, true)
    @test !Gen.has_output_grad(d)
end

@testitem "Site conditional matches the AutoGP Gaussian predictive" begin
    using AutoGP
    using NowcastAutoGP: _site_conditional
    Distributions = AutoGP.Distributions

    ts = collect(range(0, 1, length = 15))
    ts_pred = [0.5, 1.05, 1.1, 1.2]
    xs = sin.(7 .* ts)
    noise = 0.02
    kernel = AutoGP.GP.Plus(
        AutoGP.GP.GammaExponential(0.3, 1.5, 0.8), AutoGP.GP.Periodic(0.5, 0.25, 0.4)
    )

    # With no extra per-point noise, the conditional is the standard GP predictive.
    mu, Sigma = _site_conditional(kernel, noise, ts, xs, zeros(length(ts)), ts_pred)
    reference = Distributions.MvNormal(kernel, noise, ts, xs, ts_pred)
    @test isapprox(mu, Distributions.mean(reference); atol = 1.0e-10)
    @test isapprox(Sigma, Distributions.cov(reference); atol = 1.0e-10)

    # Extra per-point noise shrinks the mean towards zero and widens the predictive.
    v = fill(0.5, length(ts))
    mu_noisy, Sigma_noisy = _site_conditional(kernel, noise, ts, xs, v, ts_pred)
    @test sum(abs2, mu_noisy) < sum(abs2, mu)
    @test all(Sigma_noisy[i, i] > Sigma[i, i] for i in eachindex(ts_pred))
end

@testitem "Fit and forecast counts" setup = [CountObservationData] begin
    using AutoGP

    # The fitted model carries the observation model and the raw counts.
    @test count_model isa GPModel
    @test count_model.config.observation_model isa NowcastAutoGP.PoissonFixedSites
    @test count_model.y_transform == AutoGP.Transforms.LinearTransform(1.0, 0.0)
    @test count_model.y == Float64.(counts)
    # Fitting shuffles the order in which data is added, so compare as sets of values.
    @test all(
        sort(t[:xs]) == sort(Float64.(counts)) for t in count_model.pf_state.traces
    )

    forecasts = forecast(count_model, count_forecast_dates, 50)
    @test forecasts isa Matrix{Int}
    @test size(forecasts) == (length(count_forecast_dates), 50)
    @test all(>=(0), forecasts)

    # Forecasts stay on the scale of the data.
    @test 0.1 * maximum(counts) < sum(forecasts) / length(forecasts) < 10 * maximum(counts)

    # A range of dates and HMC refinement between draws both work.
    forecasts_hmc = forecast(
        count_model, count_dates[end] .+ Week.(1:2), 4; forecast_n_hmc = 1
    )
    @test forecasts_hmc isa Matrix{Int}
    @test size(forecasts_hmc) == (2, 4)

    # AutoGP's Gaussian predictive is not available for a count model.
    @test_throws ErrorException AutoGP.predict_mvn(count_model, collect(count_forecast_dates))
end

@testitem "forecast_with_nowcasts with counts" setup = [CountObservationData] begin
    nowcast_dates = [count_dates[end] + Week(1), count_dates[end] + Week(2)]
    forecast_dates = count_dates[end] .+ Week.(3:5)
    n_before = length(count_model.ds)

    # Nowcast scenarios are counts and need not be integers.
    nowcasts = create_nowcast_data([[3, 4], [5, 2], [4.5, 3.2]], nowcast_dates)
    forecasts = forecast_with_nowcasts(count_model, nowcasts, forecast_dates, 20)
    @test forecasts isa Matrix{Int}
    @test size(forecasts) == (length(forecast_dates), 3 * 20)
    @test all(>=(0), forecasts)
    @test length(count_model.ds) == n_before  # base model is not mutated

    # Refining parameters after adding each nowcast also works.
    forecasts_hmc = forecast_with_nowcasts(
        count_model, nowcasts[1:2], forecast_dates, 5; n_hmc = 1
    )
    @test size(forecasts_hmc) == (length(forecast_dates), 2 * 5)

    # Transformed (here negative) nowcast values are rejected with a clear error.
    bad_nowcasts = create_nowcast_data([[3, 4]], nowcast_dates; transformation = y -> -y)
    @test_throws ArgumentError forecast_with_nowcasts(
        count_model, bad_nowcasts, forecast_dates, 5
    )
end

@testitem "Count model with constant low rate" begin
    using Dates
    using Random
    Random.seed!(11)

    # A flat rate of 4 per week: forecasts should centre near it.
    dates = collect(Date(2023, 1, 7):Week(1):Date(2023, 12, 30))
    counts = NowcastAutoGP._sample_counts(fill(log(4.0), length(dates)))
    model = make_and_fit_model(
        create_count_data(dates, counts); n_particles = 4, n_mcmc = 20, n_hmc = 10
    )
    forecasts = forecast(model, dates[end] .+ Week.(1:2), 400)
    @test 1.5 < sum(forecasts) / length(forecasts) < 10

    # An all-zero series fits and forecasts small counts.
    zero_model = make_and_fit_model(
        create_count_data(dates, zeros(Int, length(dates)));
        n_particles = 2, n_mcmc = 10, n_hmc = 5
    )
    zero_forecasts = forecast(zero_model, dates[end] .+ Week.(1:2), 400)
    @test sum(zero_forecasts) / length(zero_forecasts) < 2
end

@testitem "make_and_fit_model rejects a preset observation model" begin
    using Dates
    dates = collect(Date(2024, 1, 6):Week(1):Date(2024, 3, 30))
    data = create_count_data(dates, collect(1:length(dates)))
    config = GPConfig(observation_model = NowcastAutoGP.PoissonFixedSites(1.0, 1.0))
    @test_throws ArgumentError make_and_fit_model(
        data; config = config, n_particles = 1, n_mcmc = 5, n_hmc = 3
    )
end
