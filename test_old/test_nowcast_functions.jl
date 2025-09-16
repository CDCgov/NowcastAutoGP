@testset "Nowcast Functions Tests" begin
    using DataFramesMeta
    using Dates
    using LogExpFunctions: logit, logistic

    @testset "create_nowcast_data" begin
        # Test data setup
        dates = [Date(2024, 1, 1), Date(2024, 1, 2), Date(2024, 1, 3)]

        @testset "Vector input method" begin
            # Test with vector of vectors
            nowcasts_vec = [[10.0, 11.0, 12.0], [9.5, 10.8, 11.2], [10.2, 11.1, 12.1]]

            result = create_nowcast_data(nowcasts_vec, dates)

            @test length(result) == 3
            @test all(r -> r.ds == dates, result)
            @test result[1].y == [10.0, 11.0, 12.0]
            @test result[1].values == [10.0, 11.0, 12.0]
            @test result[2].y == [9.5, 10.8, 11.2]
            @test result[3].y == [10.2, 11.1, 12.1]
        end

        @testset "Matrix input method" begin
            # Test with matrix (columns = scenarios)
            nowcasts_matrix = [10.0 9.5 10.2;
                               11.0 10.8 11.1;
                               12.0 11.2 12.1]

            result = create_nowcast_data(nowcasts_matrix, dates)

            @test length(result) == 3
            @test all(r -> r.ds == dates, result)
            @test result[1].y == [10.0, 11.0, 12.0]
            @test result[2].y == [9.5, 10.8, 11.2]
            @test result[3].y == [10.2, 11.1, 12.1]

            # Verify both methods produce identical results
            nowcasts_vec = [[10.0, 11.0, 12.0], [9.5, 10.8, 11.2], [10.2, 11.1, 12.1]]
            result_vec = create_nowcast_data(nowcasts_vec, dates)

            @test length(result) == length(result_vec)
            for i in 1:length(result)
                @test result[i].ds == result_vec[i].ds
                @test result[i].y == result_vec[i].y
                @test result[i].values == result_vec[i].values
            end
        end

        @testset "Transform function" begin
            nowcasts_vec = [[1.0, 2.0], [1.5, 2.5]]
            dates_short = [Date(2024, 1, 1), Date(2024, 1, 2)]

            # Test with log transform
            result = create_nowcast_data(nowcasts_vec, dates_short; transformation = log)

            @test result[1].y ≈ [log(1.0), log(2.0)]
            @test result[2].y ≈ [log(1.5), log(2.5)]
            @test result[1].values == [1.0, 2.0]  # Original values preserved
            @test result[2].values == [1.5, 2.5]
        end

        @testset "Error conditions" begin
            dates_short = [Date(2024, 1, 1), Date(2024, 1, 2)]

            # Empty nowcasts
            @test_throws AssertionError create_nowcast_data(
                Vector{Vector{Float64}}(), dates_short)

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

        @testset "Edge cases" begin
            # Single date, single scenario
            single_date = [Date(2024, 1, 1)]
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

        @testset "NamedTuple structure" begin
            nowcasts_vec = [[10.0, 11.0]]
            dates_short = [Date(2024, 1, 1), Date(2024, 1, 2)]

            result = create_nowcast_data(nowcasts_vec, dates_short)

            # Check NamedTuple has correct fields
            @test haskey(result[1], :ds)
            @test haskey(result[1], :y)
            @test haskey(result[1], :values)

            # Check field types
            @test result[1].ds isa Vector{Date}
            @test result[1].y isa Vector{Float64}
            @test result[1].values isa Vector{Float64}
        end
    end

    @testset "forecast_with_nowcasts" begin
        # Setup base data and model
        dates = Date(2024, 1, 1):Day(1):Date(2024, 1, 10)
        values = [10.0, 15.0, 12.0, 18.0, 22.0, 25.0, 20.0, 16.0, 14.0, 11.0]
        df = DataFrame(date = collect(dates), value = values)

        # Create and fit base model
        base_data = create_transformed_data(df)
        base_model = make_and_fit_model(base_data; n_particles = 1, n_mcmc = 5, n_hmc = 5)

        @testset "Basic functionality" begin
            # Create nowcast scenarios
            nowcast_dates = [Date(2024, 1, 11), Date(2024, 1, 12)]
            nowcast_scenarios = [
                (ds = nowcast_dates, y = [12.0, 13.0], values = [12.0, 13.0]),
                (ds = nowcast_dates, y = [11.5, 12.8], values = [11.5, 12.8])
            ]

            forecast_dates = [Date(2024, 1, 13), Date(2024, 1, 14)]
            forecast_draws_per_nowcast = 10

            result = forecast_with_nowcasts(
                base_model, nowcast_scenarios, forecast_dates, forecast_draws_per_nowcast
            )

            # Check dimensions: (n_forecast_dates, n_scenarios * draws_per_scenario)
            expected_cols = length(nowcast_scenarios) * forecast_draws_per_nowcast
            @test size(result) == (length(forecast_dates), expected_cols)
            @test size(result, 1) == 2  # 2 forecast dates
            @test size(result, 2) == 20  # 2 scenarios * 10 draws each
        end

        @testset "Single nowcast scenario" begin
            nowcast_dates = [Date(2024, 1, 11)]
            single_nowcast = [
                (ds = nowcast_dates, y = [12.0], values = [12.0])
            ]

            forecast_dates = [Date(2024, 1, 12)]
            forecast_draws_per_nowcast = 5

            result = forecast_with_nowcasts(
                base_model, single_nowcast, forecast_dates, forecast_draws_per_nowcast
            )

            @test size(result) == (1, 5)  # 1 forecast date, 5 draws
        end

        @testset "Transform function application" begin
            nowcast_dates = [Date(2024, 1, 11)]
            nowcast_scenarios = [
                (ds = nowcast_dates, y = [log(12.0)], values = [12.0])  # Pre-transformed
            ]

            forecast_dates = [Date(2024, 1, 12)]
            forecast_draws_per_nowcast = 3

            # Test inverse transform (exp to undo log)
            result = forecast_with_nowcasts(
                base_model, nowcast_scenarios, forecast_dates, forecast_draws_per_nowcast;
                inv_transformation = exp
            )

            @test size(result) == (1, 3)
            @test all(result .> 0)  # Should be positive after exp transform
        end

        @testset "MCMC refinement options" begin
            nowcast_dates = [Date(2024, 1, 11)]
            nowcast_scenarios = [
                (ds = nowcast_dates, y = [12.0], values = [12.0])
            ]
            forecast_dates = [Date(2024, 1, 12)]

            # Test parameter-only updates (n_mcmc = 0, n_hmc > 0)
            result_params = forecast_with_nowcasts(
                base_model, nowcast_scenarios, forecast_dates, 2;
                n_mcmc = 0, n_hmc = 2
            )
            @test size(result_params) == (1, 2)

            # Test full MCMC (n_mcmc > 0, n_hmc > 0)
            result_full = forecast_with_nowcasts(
                base_model, nowcast_scenarios, forecast_dates, 2;
                n_mcmc = 2, n_hmc = 2
            )
            @test size(result_full) == (1, 2)

            # Test no refinement (both = 0)
            result_none = forecast_with_nowcasts(
                base_model, nowcast_scenarios, forecast_dates, 2;
                n_mcmc = 0, n_hmc = 0
            )
            @test size(result_none) == (1, 2)
        end

        @testset "Particle resampling" begin
            nowcast_dates = [Date(2024, 1, 11)]
            nowcast_scenarios = [
                (ds = nowcast_dates, y = [12.0], values = [12.0])
            ]
            forecast_dates = [Date(2024, 1, 12)]

            # Test with resampling threshold
            result = forecast_with_nowcasts(
                base_model, nowcast_scenarios, forecast_dates, 2;
                ess_threshold = 0.5  # Force resampling
            )
            @test size(result) == (1, 2)
        end

        @testset "Multiple forecast dates" begin
            nowcast_dates = [Date(2024, 1, 11)]
            nowcast_scenarios = [
                (ds = nowcast_dates, y = [12.0], values = [12.0]),
                (ds = nowcast_dates, y = [11.8], values = [11.8])
            ]

            # Test with date range
            forecast_dates = Date(2024, 1, 12):Day(1):Date(2024, 1, 15)
            forecast_draws_per_nowcast = 3

            result = forecast_with_nowcasts(
                base_model, nowcast_scenarios, forecast_dates, forecast_draws_per_nowcast
            )

            @test size(result) == (4, 6)  # 4 forecast dates, 2 scenarios * 3 draws
        end

        @testset "Error conditions" begin
            nowcast_dates = [Date(2024, 1, 11)]
            forecast_dates = [Date(2024, 1, 12)]

            # Empty nowcasts
            @test_throws AssertionError forecast_with_nowcasts(
                base_model, NamedTuple[], forecast_dates, 5
            )

            # Invalid MCMC parameters (n_mcmc > 0 but n_hmc = 0)
            nowcast_scenarios = [(ds = nowcast_dates, y = [12.0], values = [12.0])]
            @test_throws AssertionError forecast_with_nowcasts(
                base_model, nowcast_scenarios, forecast_dates, 5;
                n_mcmc = 5, n_hmc = 0
            )
        end

        @testset "Consistency checks" begin
            # Test that results are deterministic with fixed seed
            nowcast_dates = [Date(2024, 1, 11)]
            nowcast_scenarios = [
                (ds = nowcast_dates, y = [12.0], values = [12.0])
            ]
            forecast_dates = [Date(2024, 1, 12)]

            # Note: Due to randomness in GP sampling, we test structural consistency
            # rather than exact reproducibility
            result1 = forecast_with_nowcasts(
                base_model, nowcast_scenarios, forecast_dates, 5
            )
            result2 = forecast_with_nowcasts(
                base_model, nowcast_scenarios, forecast_dates, 5
            )

            @test size(result1) == size(result2)
            @test all(isfinite.(result1))
            @test all(isfinite.(result2))
        end

        @testset "Integration with create_nowcast_data" begin
            # Test full workflow: matrix -> nowcast data -> forecasts
            nowcast_matrix = [12.0 11.8; 13.0 12.5]  # 2 time points, 2 scenarios
            nowcast_dates = [Date(2024, 1, 11), Date(2024, 1, 12)]

            # Create nowcast data using utility function
            nowcast_scenarios = create_nowcast_data(nowcast_matrix, nowcast_dates)

            forecast_dates = [Date(2024, 1, 13)]
            result = forecast_with_nowcasts(
                base_model, nowcast_scenarios, forecast_dates, 3
            )

            @test size(result) == (1, 6)  # 1 forecast date, 2 scenarios * 3 draws
        end
    end
end
