@testset "Helper Functions Tests" begin
    using DataFramesMeta
    using Dates
    using LogExpFunctions: logit, logistic

    @testset "create_transformed_data" begin
        # Test data setup
        dates = Date(2024, 1, 1):Day(1):Date(2024, 1, 10)
        values = [10.0, 15.0, 12.0, 18.0, 22.0, 25.0, 20.0, 16.0, 14.0, 11.0]
        df = DataFrame(date = dates, value = values)

        @testset "Basic functionality" begin
            # Test with default parameters
            result = create_transformed_data(df)

            @test result.ds == dates
            @test result.y == values
            @test result.values == values
            @test length(result.ds) == length(dates)
            @test length(result.y) == length(values)
        end

        @testset "Custom column names" begin
            # Test with custom column names
            df_custom = DataFrame(timestamp = dates, count = values)
            result = create_transformed_data(
                df_custom; time_col = :timestamp, value_col = :count)

            @test result.ds == dates
            @test result.y == values
            @test result.values == values
        end

        @testset "Transform function" begin
            # Test with log transformation
            result = create_transformed_data(df; transformation = log)

            @test result.ds == dates
            @test result.y ≈ log.(values)
            @test result.values == values  # Original values should remain unchanged
        end

        @testset "Logit transformation" begin
            # Test with logit transformation (need values between 0 and 1)
            proportions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
            df_prop = DataFrame(date = dates, value = proportions)
            result = create_transformed_data(df_prop; transformation = logit)

            @test result.ds == dates
            @test result.y ≈ logit.(proportions)
            @test result.values == proportions
        end

        @testset "Data redaction" begin
            # Test with n_redact parameter
            n_redact = 3
            result = create_transformed_data(df; n_redact = n_redact)

            expected_length = length(dates) - n_redact
            @test length(result.ds) == expected_length
            @test length(result.y) == expected_length
            @test length(result.values) == expected_length
            @test result.ds == dates[1:expected_length]
            @test result.y == values[1:expected_length]
        end

        @testset "Edge cases" begin
            # Test with minimum data (1 row)
            df_min = DataFrame(date = [Date(2024, 1, 1)], value = [10.0])
            result = create_transformed_data(df_min)
            @test length(result.ds) == 1
            @test length(result.y) == 1

            # Test error when n_redact is too large
            @test_throws ErrorException create_transformed_data(
                df; n_redact = length(dates))
            @test_throws ErrorException create_transformed_data(
                df; n_redact = length(dates) + 1)
        end

        @testset "Transform with redaction" begin
            # Test combining transformation and redaction
            n_redact = 2
            result = create_transformed_data(df; transformation = sqrt, n_redact = n_redact)

            expected_length = length(dates) - n_redact
            @test length(result.ds) == expected_length
            @test result.y ≈ sqrt.(values[1:expected_length])
            @test result.values == values[1:expected_length]
        end
    end
end
