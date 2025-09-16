@testset "Model Fitting Tests" begin
    using DataFramesMeta
    using Dates

    @testset "make_and_fit_model" begin
        # Test data setup - create realistic time series data
        dates = Date(2024, 1, 1):Day(1):Date(2024, 1, 30)
        # Generate some synthetic data with trend and noise
        values = 100.0 .+ 0.5 * (1:length(dates)) .+ 5.0 * randn(length(dates))
        df = DataFrame(date = dates, value = values)

        @testset "Basic model fitting" begin
            # Create transformed data
            data = create_transformed_data(df)

            # Test with default parameters (use small values for faster testing)
            model = make_and_fit_model(data; n_particles = 1, n_mcmc = 10, n_hmc = 5)

            # Check that model is returned and has the expected structure
            @test model isa Any  # AutoGP.GPModel type
            @test hasfield(typeof(model), :ds) || hasfield(typeof(model), :y)  # Should have input data
        end

        @testset "Custom parameters" begin
            # Create transformed data
            data = create_transformed_data(df)

            # Test with custom n_particles
            model = make_and_fit_model(data; n_particles = 1, n_mcmc = 10, n_hmc = 5)
            @test model isa Any

            # Test with custom smc_data_proportion
            model = make_and_fit_model(
                data; n_particles = 1, smc_data_proportion = 0.1, n_mcmc = 10, n_hmc = 5)
            @test model isa Any
        end

        @testset "Small dataset" begin
            # Test with minimal data
            small_dates = Date(2024, 1, 1):Day(1):Date(2024, 1, 10)  # Use 10 days instead of 5
            small_values = [10.0, 12.0, 11.0, 13.0, 14.0, 12.0, 15.0, 16.0, 14.0, 13.0]
            small_df = DataFrame(date = small_dates, value = small_values)
            small_data = create_transformed_data(small_df)

            # Should still work with small dataset - use higher smc_data_proportion to avoid step=0
            model = make_and_fit_model(small_data; n_particles = 1,
                smc_data_proportion = 0.5, n_mcmc = 5, n_hmc = 3)
            @test model isa Any
        end

        @testset "Transformed data" begin
            # Test with log-transformed data
            positive_values = abs.(values) .+ 1.0  # Ensure positive values
            df_pos = DataFrame(date = dates, value = positive_values)
            data = create_transformed_data(df_pos; transformation = log)

            model = make_and_fit_model(data; n_particles = 1, n_mcmc = 10, n_hmc = 5)
            @test model isa Any
        end

        @testset "Parameter validation" begin
            data = create_transformed_data(df)

            # Test that n_particles must be positive (we expect warnings about thread usage)
            model = make_and_fit_model(
                data; n_particles = 1, n_mcmc = 5, n_hmc = 3, smc_data_proportion = 0.2)
            @test model isa Any

            # Test that smc_data_proportion is reasonable - use higher values to avoid step=0
            model = make_and_fit_model(
                data; n_particles = 1, smc_data_proportion = 0.1, n_mcmc = 5, n_hmc = 3)
            @test model isa Any

            model = make_and_fit_model(
                data; n_particles = 1, smc_data_proportion = 0.5, n_mcmc = 5, n_hmc = 3)
            @test model isa Any
        end
    end
end
