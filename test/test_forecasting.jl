@testsnippet ForecastingData begin
    using DataFramesMeta, Dates, LogExpFunctions
    using Random
    Random.seed!(123)  # For reproducible test data

    # Main test data - realistic time series
    dates = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 20))
    values = 100.0 .+ 0.5 * (1:length(dates)) .+ 2.0 * randn(length(dates))

    # Proportion data for logit transformation testing
    prop_values = 0.1 .+ 0.8 * (1:length(dates)) / length(dates) .+
                  0.05 * randn(length(dates))
    prop_values = clamp.(prop_values, 0.01, 0.99)  # Keep in valid range

    # Test parameters for fast model fitting
    test_params = (n_particles = 1, n_mcmc = 10, n_hmc = 5)
    minimal_params = (n_particles = 1, n_mcmc = 5, n_hmc = 3)

    # Forecast dates for testing
    forecast_dates = collect(Date(2024, 1, 21):Day(1):Date(2024, 1, 25))
    short_dates = [Date(2024, 1, 21)]
    long_dates = collect(Date(2024, 1, 21):Day(1):Date(2024, 1, 30))

    # Pre-fitted models for reuse in tests
    data = create_transformed_data(dates, values; transformation = identity)
    model = make_and_fit_model(data; test_params...)

    data_logit = create_transformed_data(dates, prop_values; transformation = logit)
    model_logit = make_and_fit_model(data_logit; minimal_params...)
end

@testitem "forecast Basic Functionality" setup=[ForecastingData] begin
    # Generate forecasts
    forecasts = forecast(model, forecast_dates, 10)

    # Check dimensions
    @test size(forecasts) == (length(forecast_dates), 10)
    @test eltype(forecasts) <: Real
end

@testitem "forecast Different Horizons" setup=[ForecastingData] begin
    # Test short horizon
    forecasts_short = forecast(model, short_dates, 5)
    @test size(forecasts_short) == (1, 5)

    # Test longer horizon
    forecasts_long = forecast(model, long_dates, 5)
    @test size(forecasts_long) == (length(long_dates), 5)
end

@testitem "forecast Different Draw Counts" setup=[ForecastingData] begin
    test_dates = forecast_dates[1:3]

    # Test with few draws
    forecasts_few = forecast(model, test_dates, 2)
    @test size(forecasts_few, 2) == 2

    # Test with many draws
    forecasts_many = forecast(model, test_dates, 100)
    @test size(forecasts_many, 2) == 100
end

@testitem "forecast Inverse Transformation" setup=[ForecastingData] begin
    test_dates = forecast_dates[1:2]

    # Test with identity transformation (default)
    forecasts_identity = forecast(model, test_dates, 5)

    # Test with custom inverse transformation
    forecasts_exp = forecast(model, test_dates, 5; inv_transformation = exp)

    # Should have same dimensions but different values
    @test size(forecasts_identity) == size(forecasts_exp)
    # Values should be positive after exp transform
    @test all(forecasts_exp .> 0)
end

@testitem "forecast Logistic Transformation" setup=[ForecastingData] begin
    # Apply logistic inverse transformation to logit-transformed model
    forecasts_logistic = forecast(
        model_logit, short_dates, 5; inv_transformation = logistic)

    # All forecasts should be between 0 and 1 after logistic transformation
    @test all(0 .< forecasts_logistic .< 1)
end

@testitem "forecast Edge Cases" setup=[ForecastingData] begin
    # Test with single draw
    forecasts_single = forecast(model, short_dates, 1)
    @test size(forecasts_single) == (1, 1)

    # Test with single date, multiple draws
    forecasts_single_date = forecast(model, short_dates, 10)
    @test size(forecasts_single_date) == (1, 10)
end

@testitem "forecast Consistency" setup=[ForecastingData] begin
    test_dates = forecast_dates[1:2]

    # Multiple runs should give different results (stochastic)
    forecasts1 = forecast(model, test_dates, 5)
    forecasts2 = forecast(model, test_dates, 5)

    # Should have same dimensions
    @test size(forecasts1) == size(forecasts2)
    @test size(forecasts1) == (2, 5)
end
