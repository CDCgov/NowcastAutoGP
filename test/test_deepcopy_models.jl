@testsnippet DeepCopyData begin
    using Dates, Random
    using AutoGP  # Add explicit import for AutoGP
    Random.seed!(42)  # For reproducible test data

    # Test data for deepcopy experiments
    dates = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 15))
    values = 100.0 .+ 0.5 * (1:length(dates)) .+ 2.0 * randn(length(dates))
    
    # Additional nowcast data for testing deepcopy with data addition
    nowcast_dates = collect(Date(2024, 1, 16):Day(1):Date(2024, 1, 20))
    nowcast_values = 110.0 .+ 0.5 * (16:20) .+ 2.0 * randn(length(nowcast_dates))

    # Test parameters for fast model fitting
    test_params = (n_particles = 2, n_mcmc = 5, n_hmc = 3)
end

@testitem "Model Deepcopy Basic Functionality" setup=[DeepCopyData] begin
    # Create and fit a model
    data = create_transformed_data(dates, values; transformation = identity)
    model = make_and_fit_model(data; test_params...)
    
    # Test that model can be deepcopied
    model_copy = deepcopy(model)
    
    # Verify that both models exist and are distinct objects
    @test model_copy isa AutoGP.GPModel
    @test model !== model_copy  # Different objects
    
    # Verify that core properties are preserved
    @test AutoGP.num_particles(model) == AutoGP.num_particles(model_copy)
end

@testitem "Deepcopy Model Can Add Data" setup=[DeepCopyData] begin
    # Create and fit a model
    data = create_transformed_data(dates, values; transformation = identity)
    model = make_and_fit_model(data; test_params...)
    
    # Create a deepcopy
    model_copy = deepcopy(model)
    
    # Create nowcast data
    nowcast_data = create_transformed_data(nowcast_dates, nowcast_values; transformation = identity)
    
    # Test that the deepcopied model can add new data
    @test_nowarn AutoGP.add_data!(model_copy, nowcast_data.ds, nowcast_data.y)
    
    # Test that we can remove the added data
    @test_nowarn AutoGP.remove_data!(model_copy, nowcast_data.ds)
    
    # Verify original model is unchanged by operations on the copy
    @test length(model.ds) == length(dates)
end

@testitem "Deepcopy vs Original Model Independence" setup=[DeepCopyData] begin
    # Create and fit a model
    data = create_transformed_data(dates, values; transformation = identity)
    model = make_and_fit_model(data; test_params...)
    
    # Create a deepcopy
    model_copy = deepcopy(model)
    
    # Get initial data sizes by checking model fields
    original_data_size = length(model.ds)
    copy_data_size = length(model_copy.ds)
    @test original_data_size == copy_data_size
    
    # Add data to the copy
    nowcast_data = create_transformed_data(nowcast_dates, nowcast_values; transformation = identity)
    AutoGP.add_data!(model_copy, nowcast_data.ds, nowcast_data.y)
    
    # Verify original model is unchanged
    @test length(model.ds) == original_data_size
    @test length(model_copy.ds) == original_data_size + length(nowcast_dates)
    
    # Clean up the copy
    AutoGP.remove_data!(model_copy, nowcast_data.ds)
    @test length(model_copy.ds) == original_data_size
end

@testitem "Deepcopy Model Forecasting Works" setup=[DeepCopyData] begin
    # Create and fit a model
    data = create_transformed_data(dates, values; transformation = identity)
    model = make_and_fit_model(data; test_params...)
    
    # Create a deepcopy
    model_copy = deepcopy(model)
    
    # Test that both models can generate forecasts
    forecast_dates = collect(Date(2024, 1, 21):Day(1):Date(2024, 1, 25))
    
    forecasts_original = forecast(model, forecast_dates, 10)
    forecasts_copy = forecast(model_copy, forecast_dates, 10)
    
    # Verify forecast dimensions
    @test size(forecasts_original) == (length(forecast_dates), 10)
    @test size(forecasts_copy) == (length(forecast_dates), 10)
    
    # Forecasts should be stochastic, so they may differ
    @test size(forecasts_original) == size(forecasts_copy)
end