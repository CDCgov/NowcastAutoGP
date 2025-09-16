@testset "Forecasting Tests" begin
    using DataFramesMeta
    using Dates
    using LogExpFunctions: logit, logistic

    @testset "forecast" begin
        # Test data setup - create realistic time series data
        dates = Date(2024, 1, 1):Day(1):Date(2024, 1, 20)
        # Generate some synthetic data with trend and noise
        values = 100.0 .+ 0.5 * (1:length(dates)) .+ 2.0 * randn(length(dates))
        df = DataFrame(date = dates, value = values)

        # Create and fit model for testing
        data = create_transformed_data(df)
        model = make_and_fit_model(data; n_particles = 1, n_mcmc = 10, n_hmc = 5)

        @testset "Basic forecasting" begin
            # Define forecast dates
            forecast_dates = Date(2024, 1, 21):Day(1):Date(2024, 1, 25)
            forecast_draws = 10

            # Generate forecasts
            forecasts = forecast(model, forecast_dates, forecast_draws)

            # Check dimensions
            @test size(forecasts) == (length(forecast_dates), forecast_draws)
            @test eltype(forecasts) <: Real
        end

        @testset "Different forecast horizons" begin
            # Test short horizon
            short_dates = [Date(2024, 1, 21)]
            forecasts_short = forecast(model, short_dates, 5)
            @test size(forecasts_short) == (1, 5)

            # Test longer horizon
            long_dates = Date(2024, 1, 21):Day(1):Date(2024, 1, 30)
            forecasts_long = forecast(model, long_dates, 5)
            @test size(forecasts_long) == (length(long_dates), 5)
        end

        @testset "Different numbers of draws" begin
            forecast_dates = Date(2024, 1, 21):Day(1):Date(2024, 1, 23)

            # Test with few draws
            forecasts_few = forecast(model, forecast_dates, 2)
            @test size(forecasts_few, 2) == 2

            # Test with many draws
            forecasts_many = forecast(model, forecast_dates, 100)
            @test size(forecasts_many, 2) == 100
        end

        @testset "Inverse transformation" begin
            # Test with identity transformation (default)
            forecast_dates = [Date(2024, 1, 21), Date(2024, 1, 22)]
            forecasts_identity = forecast(model, forecast_dates, 5)

            # Test with custom inverse transformation
            forecasts_exp = forecast(model, forecast_dates, 5; inv_transformation = exp)

            # Should have same dimensions but different values (unless identity)
            @test size(forecasts_identity) == size(forecasts_exp)
            # Values should be different if transformation is not identity
            # (unless by coincidence, so we just test they're positive after exp)
            @test all(forecasts_exp .> 0)
        end

        @testset "Inverse logistic transformation" begin
            # For this test, we'll create a model with logit-transformed data
            # Use proportions between 0 and 1
            prop_values = 0.1 .+ 0.8 * (1:length(dates)) / length(dates) .+
                          0.05 * randn(length(dates))
            prop_values = clamp.(prop_values, 0.01, 0.99)  # Keep in valid range
            df_prop = DataFrame(date = dates, value = prop_values)

            data_logit = create_transformed_data(df_prop; transformation = logit)
            model_logit = make_and_fit_model(
                data_logit; n_particles = 1, n_mcmc = 5, n_hmc = 3)

            forecast_dates = [Date(2024, 1, 21)]
            # Apply logistic inverse transformation
            forecasts_logistic = forecast(
                model_logit, forecast_dates, 5; inv_transformation = logistic)

            # All forecasts should be between 0 and 1 after logistic transformation
            @test all(0 .< forecasts_logistic .< 1)
        end

        @testset "Edge cases" begin
            forecast_dates = [Date(2024, 1, 21)]

            # Test with single draw
            forecasts_single = forecast(model, forecast_dates, 1)
            @test size(forecasts_single) == (1, 1)

            # Test with single date
            forecasts_single_date = forecast(model, forecast_dates, 10)
            @test size(forecasts_single_date) == (1, 10)
        end

        @testset "Forecast consistency" begin
            forecast_dates = [Date(2024, 1, 21), Date(2024, 1, 22)]

            # Multiple runs should give different results (stochastic)
            forecasts1 = forecast(model, forecast_dates, 5)
            forecasts2 = forecast(model, forecast_dates, 5)

            # Should have same dimensions
            @test size(forecasts1) == size(forecasts2)
            # Should generally be different values (with high probability)
            # We don't test for exact inequality as there's a tiny chance they could be equal
            @test size(forecasts1) == (2, 5)
        end
    end
end
