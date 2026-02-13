@testsnippet NowcastData begin
    using Dates, LogExpFunctions
    using Random
    Random.seed!(123)  # For reproducible test data

    # Basic test dates
    dates = [Date(2024, 1, 1), Date(2024, 1, 2), Date(2024, 1, 3)]
    dates_short = [Date(2024, 1, 1), Date(2024, 1, 2)]
    single_date = [Date(2024, 1, 1)]

    # Test data for create_nowcast_data
    nowcasts_vec = [[10.0, 11.0, 12.0], [9.5, 10.8, 11.2], [10.2, 11.1, 12.1]]
    nowcasts_matrix = [10.0 9.5 10.2;
                       11.0 10.8 11.1;
                       12.0 11.2 12.1]

    # Data for transformation testing
    transform_vec = [[1.0, 2.0], [1.5, 2.5]]
end

@testsnippet ForecastingWithNowcastsData begin
    using Dates
    using Random
    Random.seed!(123)

    # Base model setup
    dates = Date(2024, 1, 1):Day(1):Date(2024, 1, 10)
    values = [10.0, 15.0, 12.0, 18.0, 22.0, 25.0, 20.0, 16.0, 14.0, 11.0]

    # Create and fit base model with minimal parameters for speed
    base_data = create_transformed_data(dates, values; transformation = x -> x)
    base_model = make_and_fit_model(base_data; n_particles = 1, n_mcmc = 5, n_hmc = 5)

    # Common test scenarios
    nowcast_dates = [Date(2024, 1, 11), Date(2024, 1, 12)]
    nowcast_scenarios_multi = [
        TData(nowcast_dates, [12.0, 13.0]; transformation = x -> x),
        TData(nowcast_dates, [11.5, 12.8]; transformation = x -> x)
    ]

    single_nowcast_dates = [Date(2024, 1, 11)]
    single_nowcast = [TData(single_nowcast_dates, [12.0]; transformation = x -> x)]

    # Forecast dates
    forecast_dates_multi = [Date(2024, 1, 13), Date(2024, 1, 14)]
    forecast_dates_single = [Date(2024, 1, 12)]
    forecast_dates_range = Date(2024, 1, 12):Day(1):Date(2024, 1, 15)
end

@testitem "create_nowcast_data Vector Input Method" setup = [NowcastData] begin
    result = create_nowcast_data(nowcasts_vec, dates)

    @test length(result) == 3
    @test all(r -> r.ds == dates, result)
    @test result[1].y == [10.0, 11.0, 12.0]
    @test result[1].values == [10.0, 11.0, 12.0]
    @test result[2].y == [9.5, 10.8, 11.2]
    @test result[3].y == [10.2, 11.1, 12.1]
end

@testitem "create_nowcast_data Matrix Input Method" setup = [NowcastData] begin
    result = create_nowcast_data(nowcasts_matrix, dates)

    @test length(result) == 3
    @test all(r -> r.ds == dates, result)
    @test result[1].y == [10.0, 11.0, 12.0]
    @test result[2].y == [9.5, 10.8, 11.2]
    @test result[3].y == [10.2, 11.1, 12.1]

    # Verify both methods produce identical results
    result_vec = create_nowcast_data(nowcasts_vec, dates)
    @test length(result) == length(result_vec)
    for i in 1:length(result)
        @test result[i].ds == result_vec[i].ds
        @test result[i].y == result_vec[i].y
        @test result[i].values == result_vec[i].values
    end
end

@testitem "create_nowcast_data Transform Function" setup = [NowcastData] begin
    # Test with log transform
    result = create_nowcast_data(transform_vec, dates_short; transformation = log)

    @test result[1].y ≈ [log(1.0), log(2.0)]
    @test result[2].y ≈ [log(1.5), log(2.5)]
    @test result[1].values == [1.0, 2.0]  # Original values preserved
    @test result[2].values == [1.5, 2.5]
end

@testitem "create_nowcast_data Error Conditions" setup = [NowcastData] begin
    # Empty nowcasts
    @test_throws AssertionError create_nowcast_data(Vector{Vector{Float64}}(), dates_short)

    # Mismatched lengths - dates vs nowcast length
    wrong_length_vec = [[10.0, 11.0, 12.0]]  # 3 elements but only 2 dates
    @test_throws AssertionError create_nowcast_data(wrong_length_vec, dates_short)

    # Different length vectors
    inconsistent_vec = [[10.0, 11.0], [9.5, 10.8, 11.2]]  # Different lengths
    @test_throws AssertionError create_nowcast_data(inconsistent_vec, dates)

    # Matrix with wrong dimensions
    wrong_matrix = [10.0 9.5; 11.0 10.8]  # 2 rows but 3 dates
    @test_throws AssertionError create_nowcast_data(wrong_matrix, dates)
end

@testitem "create_nowcast_data Edge Cases" setup = [NowcastData] begin
    # Single date, single scenario
    single_nowcast = [[5.0]]
    result = create_nowcast_data(single_nowcast, single_date)
    @test length(result) == 1
    @test result[1].ds == single_date
    @test result[1].y == [5.0]

    # Single date, multiple scenarios
    multi_scenario = [[5.0], [4.5], [5.2]]
    result = create_nowcast_data(multi_scenario, single_date)
    @test length(result) == 3
    @test all(r -> r.ds == single_date, result)
    @test result[1].y == [5.0]
    @test result[2].y == [4.5]
    @test result[3].y == [5.2]
end

@testitem "create_nowcast_data TData Structure" setup = [NowcastData] begin
    nowcasts_vec = [[10.0, 11.0]]
    result = create_nowcast_data(nowcasts_vec, dates_short)

    # Check TData has correct fields
    @test hasproperty(result[1], :ds)
    @test hasproperty(result[1], :y)
    @test hasproperty(result[1], :values)

    # Check field types
    @test result[1].ds isa Vector{Date}
    @test result[1].y isa Vector{Float64}
    @test result[1].values isa Vector{Float64}
end

@testitem "forecast_with_nowcasts Basic Functionality" setup = [ForecastingWithNowcastsData] begin
    forecast_draws_per_nowcast = 10
    result = forecast_with_nowcasts(
        base_model, nowcast_scenarios_multi, forecast_dates_multi, forecast_draws_per_nowcast
    )

    # Check dimensions: (n_forecast_dates, n_scenarios * draws_per_scenario)
    expected_cols = length(nowcast_scenarios_multi) * forecast_draws_per_nowcast
    @test size(result) == (length(forecast_dates_multi), expected_cols)
    @test size(result, 1) == 2  # 2 forecast dates
    @test size(result, 2) == 20  # 2 scenarios * 10 draws each
end

@testitem "forecast_with_nowcasts Single Nowcast Scenario" setup = [ForecastingWithNowcastsData] begin
    forecast_draws_per_nowcast = 5
    result = forecast_with_nowcasts(
        base_model, single_nowcast, forecast_dates_single, forecast_draws_per_nowcast
    )

    @test size(result) == (1, 5)  # 1 forecast date, 5 draws
end

@testitem "forecast_with_nowcasts Transform Function Application" setup = [ForecastingWithNowcastsData] begin
    # Pre-transformed scenario
    nowcast_scenarios = [TData(single_nowcast_dates, [log(12.0)]; transformation = x -> x)]
    forecast_draws_per_nowcast = 3

    # Test inverse transform (exp to undo log)
    result = forecast_with_nowcasts(
        base_model, nowcast_scenarios, forecast_dates_single, forecast_draws_per_nowcast;
        inv_transformation = exp
    )

    @test size(result) == (1, 3)
    @test all(result .> 0)  # Should be positive after exp transform
end

@testitem "forecast_with_nowcasts MCMC Refinement Options" setup = [ForecastingWithNowcastsData] begin
    # Test parameter-only updates (n_mcmc = 0, n_hmc > 0)
    result_params = forecast_with_nowcasts(
        base_model, single_nowcast, forecast_dates_single, 2;
        n_mcmc = 0, n_hmc = 2
    )
    @test size(result_params) == (1, 2)

    # Test full MCMC (n_mcmc > 0, n_hmc > 0)
    result_full = forecast_with_nowcasts(
        base_model, single_nowcast, forecast_dates_single, 2;
        n_mcmc = 2, n_hmc = 2
    )
    @test size(result_full) == (1, 2)

    # Test no refinement (both = 0)
    result_none = forecast_with_nowcasts(
        base_model, single_nowcast, forecast_dates_single, 2;
        n_mcmc = 0, n_hmc = 0
    )
    @test size(result_none) == (1, 2)
end

@testitem "forecast_with_nowcasts Particle Resampling" setup = [ForecastingWithNowcastsData] begin
    # Test with resampling threshold
    result = forecast_with_nowcasts(
        base_model, single_nowcast, forecast_dates_single, 2;
        ess_threshold = 0.5  # Force resampling
    )
    @test size(result) == (1, 2)
end

@testitem "forecast_with_nowcasts Multiple Forecast Dates" setup = [ForecastingWithNowcastsData] begin
    nowcast_scenarios = [
        TData(single_nowcast_dates, [12.0]; transformation = x -> x),
        TData(single_nowcast_dates, [11.8]; transformation = x -> x)
    ]
    forecast_draws_per_nowcast = 3

    result = forecast_with_nowcasts(
        base_model, nowcast_scenarios, forecast_dates_range, forecast_draws_per_nowcast
    )

    @test size(result) == (4, 6)  # 4 forecast dates, 2 scenarios * 3 draws
end

@testitem "forecast_with_nowcasts Error Conditions" setup = [ForecastingWithNowcastsData] begin
    # Empty nowcasts
    @test_throws AssertionError forecast_with_nowcasts(
        base_model, TData[], forecast_dates_single, 5
    )

    # Invalid MCMC parameters (n_mcmc > 0 but n_hmc = 0)
    @test_throws AssertionError forecast_with_nowcasts(
        base_model, single_nowcast, forecast_dates_single, 5;
        n_mcmc = 5, n_hmc = 0
    )
end

@testitem "forecast_with_nowcasts Consistency Checks" setup = [ForecastingWithNowcastsData] begin
    # Test structural consistency rather than exact reproducibility due to randomness
    result1 = forecast_with_nowcasts(base_model, single_nowcast, forecast_dates_single, 5)
    result2 = forecast_with_nowcasts(base_model, single_nowcast, forecast_dates_single, 5)

    @test size(result1) == size(result2)
    @test all(isfinite.(result1))
    @test all(isfinite.(result2))
end

@testitem "forecast_with_nowcasts Integration with create_nowcast_data" setup = [ForecastingWithNowcastsData] begin
    # Test full workflow: matrix -> nowcast data -> forecasts
    nowcast_matrix = [12.0 11.8; 13.0 12.5]  # 2 time points, 2 scenarios
    nowcast_dates = [Date(2024, 1, 11), Date(2024, 1, 12)]

    # Create nowcast data using utility function
    nowcast_scenarios = create_nowcast_data(nowcast_matrix, nowcast_dates)

    forecast_dates = [Date(2024, 1, 13)]
    result = forecast_with_nowcasts(base_model, nowcast_scenarios, forecast_dates, 3)

    @test size(result) == (1, 6)  # 1 forecast date, 2 scenarios * 3 draws
end
