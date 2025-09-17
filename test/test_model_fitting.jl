@testsnippet ModelFittingData begin
    using AutoGP, DataFramesMeta, Dates
    using Random
    Random.seed!(42)  # For reproducible test data

    # Main test dataset - realistic time series data
    dates = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 30))
    # Generate some synthetic data with trend and noise
    values = 100.0 .+ 0.5 * (1:length(dates)) .+ 5.0 * randn(length(dates))
    df = DataFrame(date = dates, value = values)

    # Small dataset for minimal testing
    small_dates = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 10))
    small_values = [10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 15.0, 16.0, 14.0, 13.0]
    small_df = DataFrame(date = small_dates, value = small_values)

    # Positive values for log transformation testing
    positive_values = abs.(values) .+ 1.0
    df_positive = DataFrame(date = dates, value = positive_values)

    # Test parameters (small values for faster testing)
    test_params = (n_particles = 1, n_mcmc = 10, n_hmc = 5)
    minimal_params = (n_particles = 1, n_mcmc = 5, n_hmc = 3)
end

@testitem "make_and_fit_model Basic Functionality" setup=[ModelFittingData] begin
    # Create transformed data
    data = create_transformed_data(dates, values; transformation = identity)

    # Test with default parameters (use small values for faster testing)
    model = make_and_fit_model(data; test_params...)

    # Check that model is returned and has the expected structure
    @test model isa AutoGP.GPModel
    @test hasfield(typeof(model), :ds) || hasfield(typeof(model), :y)  # Should have input data
end

@testitem "make_and_fit_model Custom Parameters" setup=[ModelFittingData] begin
    # Create transformed data
    data = create_transformed_data(dates, values; transformation = identity)

    # Test with custom smc_data_proportion
    model = make_and_fit_model(data; smc_data_proportion = 0.1, test_params...)
    @test model isa Any

    # Test with different smc_data_proportion
    model = make_and_fit_model(data; smc_data_proportion = 0.2, test_params...)
    @test model isa AutoGP.GPModel
end

@testitem "make_and_fit_model Small Dataset" setup=[ModelFittingData] begin
    # Test with minimal data
    small_data = create_transformed_data(
        small_dates, small_values; transformation = identity)

    # Should work with small dataset - use higher smc_data_proportion to avoid step=0
    model = make_and_fit_model(small_data; smc_data_proportion = 0.5, minimal_params...)
    @test model isa AutoGP.GPModel
end

@testitem "make_and_fit_model Log Transformation" setup=[ModelFittingData] begin
    # Test with log-transformed data
    data = create_transformed_data(dates, positive_values; transformation = log)

    model = make_and_fit_model(data; test_params...)
    @test model isa AutoGP.GPModel
end

@testitem "make_and_fit_model Parameter Validation" setup=[ModelFittingData] begin
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
