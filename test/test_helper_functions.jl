@testsnippet DataSnippet begin
    using Dates, LogExpFunctions

    # Test data setup - main datasets
    dates = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 10))
    values = [10.0, 15.0, 12.0, 18.0, 22.0, 25.0, 20.0, 16.0, 14.0, 11.0]

    # Additional test datasets
    dates_short = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 5))
    values_short = [10.0, 15.0, 12.0, 18.0, 22.0]
    int_values = [1, 2, 3, 4, 5]
    proportions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    # Test values for transformations
    test_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    positive_values = [0.1, 1.0, 5.0, 10.0, 100.0]
    percentage_values = [10.0, 25.0, 50.0, 75.0, 90.0]

    # Validation data
    short_dates = [Date(2024, 1, 1), Date(2024, 1, 2)]
    long_values = [1.0, 2.0, 3.0, 4.0]
end

@testitem "TData Basic Construction" setup=[DataSnippet] begin
    # Test with identity transformation
    result = TData(dates, values; transformation = identity)

    @test result.ds == dates
    @test result.y == values
    @test result.values == values
    @test length(result.ds) == length(dates)
    @test length(result.y) == length(values)
    @test eltype(result.y) == eltype(result.values)
end

@testitem "TData Log Transformation" setup=[DataSnippet] begin
    # Test with log transformation
    result = TData(dates, values; transformation = log)

    @test result.ds == dates
    @test result.y ≈ log.(values)
    @test result.values == values
    @test length(result.ds) == length(dates)
end

@testitem "TData Logit Transformation" setup=[DataSnippet] begin
    # Test with logit transformation (need values between 0 and 1)
    result = TData(dates, proportions; transformation = logit)

    @test result.ds == dates
    @test result.y ≈ logit.(proportions)
    @test result.values == proportions
end

@testitem "TData Type Promotion" setup=[DataSnippet] begin
    # Test automatic type promotion between different numeric types
    # Transformation that returns Float64
    result = TData(dates_short, int_values; transformation = x -> x * 1.5)

    @test eltype(result.y) == Float64
    @test eltype(result.values) == Float64
    @test result.y ≈ [1.5, 3.0, 4.5, 6.0, 7.5]
    @test result.values == [1.0, 2.0, 3.0, 4.0, 5.0]
end

@testitem "TData Length Validation" setup=[DataSnippet] begin
    # Test error when vectors have different lengths
    @test_throws AssertionError TData(short_dates, long_values; transformation = identity)
end

@testitem "TData Custom Transformations" setup=[DataSnippet] begin
    # Test with custom transformation function
    custom_transform = x -> (x - 15.0) / 5.0  # Standardization-like transform
    result = TData(dates, values; transformation = custom_transform)

    @test result.ds == dates
    @test result.y ≈ custom_transform.(values)
    @test result.values == values
end

@testitem "create_transformed_data Basic Functionality" setup=[DataSnippet] begin
    # Test convenience function with ranges and basic arrays
    result = create_transformed_data(dates_short, values_short; transformation = identity)

    @test result.ds == dates_short
    @test result.y == values_short
    @test result.values == values_short
end

@testitem "create_transformed_data With Transformation" setup=[DataSnippet] begin
    # Test convenience function with sqrt transformation
    result = create_transformed_data(dates_short, values_short; transformation = sqrt)

    @test result.ds == dates_short
    @test result.y ≈ sqrt.(values_short)
    @test result.values == values_short
end

@testitem "get_transformations Percentage" setup=[DataSnippet] begin
    forward_transform, inverse_transform = get_transformations("percentage", test_values)

    # Test that we get functions
    @test isa(forward_transform, Function)
    @test isa(inverse_transform, Function)

    # Test round-trip transformation for percentage values
    for val in percentage_values
        transformed = forward_transform(val)
        recovered = inverse_transform(transformed)
        @test recovered ≈ val atol=1e-10
    end
end

@testitem "get_transformations Positive" setup=[DataSnippet] begin
    forward_transform, inverse_transform = get_transformations("positive", positive_values)

    @test isa(forward_transform, Function)
    @test isa(inverse_transform, Function)

    # Test round-trip for positive values
    for val in positive_values
        if val > 0  # Only test positive values
            transformed = forward_transform(val)
            recovered = inverse_transform(transformed)
            @test recovered ≈ val atol=1e-6
        end
    end
end

@testitem "get_transformations BoxCox" setup=[DataSnippet] begin
    boxcox_values = [1.0, 2.0, 5.0, 10.0, 20.0]  # BoxCox requires positive values
    forward_transform, inverse_transform = get_transformations("boxcox", boxcox_values)

    @test isa(forward_transform, Function)
    @test isa(inverse_transform, Function)

    # Test round-trip for positive values (BoxCox requires positive values)
    for val in boxcox_values
        transformed = forward_transform(val)
        recovered = inverse_transform(transformed)
        @test recovered ≈ val atol=1e-6
    end
end

@testitem "get_transformations BoxCox Edge Cases" setup=[DataSnippet] begin
    # Test with very small positive values to trigger edge cases
    small_values = [1e-8, 1e-6, 1e-4, 0.001, 0.01, 0.1, 1.0, 10.0]
    forward_transform, inverse_transform = get_transformations("boxcox", small_values)

    # Test round-trip for small values
    for val in small_values
        transformed = forward_transform(val)
        recovered = inverse_transform(transformed)
        @test recovered ≈ val atol=1e-6
    end

    # Test that inverse transform handles very negative inputs gracefully
    # (these would correspond to very small lambda_y_plus_1 values)
    very_negative_inputs = [-100.0, -50.0, -20.0, -10.0]
    for input in very_negative_inputs
        result = inverse_transform(input)
        @test result ≥ 0.0  # Should never return negative values
        @test isfinite(result)  # Should always be finite
    end

    # Test that inverse transform handles very positive inputs gracefully
    very_positive_inputs = [100.0, 50.0, 20.0, 10.0]
    for input in very_positive_inputs
        result = inverse_transform(input)
        @test result ≥ 0.0  # Should never return negative values
        @test isfinite(result)  # Should always be finite
    end
end

@testitem "get_transformations BoxCox Negative Lambda" setup=[DataSnippet] begin
    # Force a scenario that might lead to negative lambda by using specific data pattern
    # Values that decrease might lead to negative lambda in Box-Cox fitting
    decreasing_values = [100.0, 50.0, 25.0, 12.5, 6.25, 3.125]
    forward_transform, inverse_transform = get_transformations("boxcox", decreasing_values)

    # Test round-trip even with potentially negative lambda
    for val in decreasing_values
        transformed = forward_transform(val)
        recovered = inverse_transform(transformed)
        @test recovered ≈ val atol=1e-4  # Slightly looser tolerance for edge cases
    end

    # Test edge case inputs that might trigger the negative lambda handling
    edge_inputs = [-5.0, -2.0, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    for input in edge_inputs
        result = inverse_transform(input)
        @test result ≥ 0.0  # Should clamp to non-negative
        @test isfinite(result)  # Should be finite
    end
end

@testitem "get_transformations BoxCox Zero Lambda Case" setup=[DataSnippet] begin
    # Test case where lambda might be very close to zero (log transformation)
    # Use values that often lead to lambda ≈ 0 in Box-Cox
    log_like_values = [1.0, 2.718, 7.389, 20.086, 54.598]  # Roughly exp(0), exp(1), etc.
    forward_transform, inverse_transform = get_transformations("boxcox", log_like_values)

    # Test that it handles inputs appropriately
    test_inputs = [-10.0, -5.0, -1.0, 0.0, 1.0, 5.0, 10.0]
    for input in test_inputs
        result = inverse_transform(input)
        @test result ≥ 0.0
        @test isfinite(result)
    end

    # Test round-trip
    for val in log_like_values
        transformed = forward_transform(val)
        recovered = inverse_transform(transformed)
        @test recovered ≈ val atol=1e-5
    end
end

@testitem "get_transformations BoxCox Numerical Stability" setup=[DataSnippet] begin
    # Test with extreme values to check numerical stability
    extreme_values = [1e-10, 1e-5, 1e-2, 1.0, 1e2, 1e5, 1e8]
    forward_transform, inverse_transform = get_transformations("boxcox", extreme_values)

    # Test that extreme transformed values don't break the inverse
    for val in extreme_values
        transformed = forward_transform(val)

        # Test the transformed value itself
        @test isfinite(transformed)

        # Test inverse
        recovered = inverse_transform(transformed)
        @test isfinite(recovered)
        @test recovered ≥ 0.0
        @test recovered ≈ val rtol=1e-3  # Relative tolerance for extreme values
    end
end

@testitem "get_transformations Unknown Error" setup=[DataSnippet] begin
    @test_throws AssertionError get_transformations("unknown", test_values)
end
