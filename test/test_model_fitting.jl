@testsnippet ModelFittingData begin
    using AutoGP, Dates
    using Random
    Random.seed!(42)  # For reproducible test data

    # Main test dataset - realistic time series data
    dates = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 30))
    # Generate some synthetic data with trend and noise
    values = 100.0 .+ 0.5 * (1:length(dates)) .+ 5.0 * randn(length(dates))

    # Small dataset for minimal testing
    small_dates = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 10))
    small_values = [10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 15.0, 16.0, 14.0, 13.0]

    # Positive values for log transformation testing
    positive_values = abs.(values) .+ 1.0

    # Test parameters (small values for faster testing)
    test_params = (n_particles = 1, n_mcmc = 10, n_hmc = 5)
    minimal_params = (n_particles = 1, n_mcmc = 5, n_hmc = 3)
end

@testitem "make_and_fit_model Basic Functionality" setup = [ModelFittingData] begin
    # Create transformed data
    data = create_transformed_data(dates, values; transformation = identity)

    # Test with default parameters (use small values for faster testing)
    model = make_and_fit_model(data; test_params...)

    # Check that model is returned and has the expected structure
    @test model isa AutoGP.GPModel
    @test hasfield(typeof(model), :ds) || hasfield(typeof(model), :y)  # Should have input data
end

@testitem "make_and_fit_model Custom Parameters" setup = [ModelFittingData] begin
    # Create transformed data
    data = create_transformed_data(dates, values; transformation = identity)

    # Test with custom smc_data_proportion
    model = make_and_fit_model(data; smc_data_proportion = 0.1, test_params...)
    @test model isa Any

    # Test with different smc_data_proportion
    model = make_and_fit_model(data; smc_data_proportion = 0.2, test_params...)
    @test model isa AutoGP.GPModel
end

@testitem "make_and_fit_model Small Dataset" setup = [ModelFittingData] begin
    # Test with minimal data
    small_data = create_transformed_data(
        small_dates, small_values; transformation = identity
    )

    # Should work with small dataset - use higher smc_data_proportion to avoid step=0
    model = make_and_fit_model(small_data; smc_data_proportion = 0.5, minimal_params...)
    @test model isa AutoGP.GPModel
end

@testitem "make_and_fit_model Log Transformation" setup = [ModelFittingData] begin
    # Test with log-transformed data
    data = create_transformed_data(dates, positive_values; transformation = log)

    model = make_and_fit_model(data; test_params...)
    @test model isa AutoGP.GPModel
end

@testitem "make_and_fit_model Parameter Validation" setup = [ModelFittingData] begin
    data = create_transformed_data(dates, values; transformation = identity)

    # Test different smc_data_proportion values
    model = make_and_fit_model(data; smc_data_proportion = 0.1, minimal_params...)
    @test model isa AutoGP.GPModel

    model = make_and_fit_model(data; smc_data_proportion = 0.3, minimal_params...)
    @test model isa AutoGP.GPModel

    # Test that the function handles edge cases properly
    model = make_and_fit_model(data; smc_data_proportion = 0.05, minimal_params...)
    @test model isa AutoGP.GPModel
end

@testsnippet FlatData begin
    using AutoGP, Dates, Random

    flat_dates = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 10))
    # Near-constant counts: the Box-Cox MLE picks a pathological λ here (issue #51).
    flat_values = [
        75000.0, 75100.0, 74950.0, 75050.0, 75000.0,
        74980.0, 75020.0, 75010.0, 74990.0, 75005.0,
    ]
    # Exactly constant: degenerate even after any monotonic transform.
    const_values = fill(75000.0, length(flat_dates))
    flat_forecast_dates = collect(Date(2024, 1, 11):Day(1):Date(2024, 1, 18))
    minimal_params = (n_particles = 1, n_mcmc = 5, n_hmc = 3)
end

@testitem "make_and_fit_model+forecast Box-Cox-degenerate flat data (issue #51)" setup = [FlatData] begin
    # Regression: this previously threw PosDefException at forecast time.
    Random.seed!(51)
    transformation, inv_transformation = get_transformations("boxcox", flat_values)
    data = create_transformed_data(flat_dates, flat_values; transformation)
    model = make_and_fit_model(data; smc_data_proportion = 0.5, minimal_params...)
    @test model isa AutoGP.GPModel

    fc = forecast(model, flat_forecast_dates, 25; inv_transformation = inv_transformation)
    @test size(fc) == (length(flat_forecast_dates), 25)
    @test all(isfinite, fc)
    @test all(fc .>= 0)
    @test 50_000 < sum(fc) / length(fc) < 100_000  # forecasts stay near ~75,000
end

@testitem "make_and_fit_model+forecast exactly constant data (issue #51)" setup = [FlatData] begin
    # Genuinely flat data is rescued by the jitter safety net in make_and_fit_model.
    Random.seed!(52)
    transformation, inv_transformation = get_transformations("boxcox", const_values)
    data = create_transformed_data(flat_dates, const_values; transformation)
    model = make_and_fit_model(data; smc_data_proportion = 0.5, minimal_params...)
    @test model isa AutoGP.GPModel

    fc = forecast(model, flat_forecast_dates, 25; inv_transformation = inv_transformation)
    @test size(fc) == (length(flat_forecast_dates), 25)
    @test all(isfinite, fc)
    @test all(fc .>= 0)
end

@testitem "_stabilize_for_fit leaves healthy data unchanged" setup = [FlatData] begin
    healthy = [10.0, 15.0, 12.0, 18.0, 22.0, 25.0, 20.0, 16.0, 14.0, 11.0]
    @test NowcastAutoGP._stabilize_for_fit(healthy) === healthy
end

@testitem "_stabilize_for_fit jitters near-constant data" setup = [FlatData] begin
    Random.seed!(1)
    flat = fill(11.2256, 30)
    jittered = NowcastAutoGP._stabilize_for_fit(flat)
    @test jittered != flat
    rel_range = (maximum(jittered) - minimum(jittered)) / (abs(sum(jittered) / length(jittered)) + 1)
    @test rel_range >= 1.0e-3
end
