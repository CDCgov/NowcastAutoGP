@testsnippet AutoGPTransformData begin
    using Dates, LogExpFunctions, AutoGP
    import BoxCox  # Needed for type assertions
    # Note: NowcastAutoGP is automatically available in TestItems context

    # Test data for different transform types
    percentage_values = [10.5, 25.3, 67.8, 89.2]
    positive_values = [1.2, 3.4, 8.9, 15.6]
    boxcox_values = [0.1, 0.5, 2.3, 5.7, 12.1]
    zero_min_values = [0.0, 1.2, 3.4, 8.9]  # Contains zero
    
    # Tolerance for round-trip error checks
    tolerance = 1e-12
end

@testitem "get_autogp_transform PercentageTransform" setup=[AutoGPTransformData] begin
    transform = get_autogp_transform("percentage", percentage_values)
    
    @test transform isa PercentageTransform{Float64}
    @test transform.offset == 0.0  # No zeros in percentage_values
    
    # Test AutoGP interface
    transformed = AutoGP.Transforms.apply(transform, percentage_values)
    recovered = AutoGP.Transforms.unapply(transform, transformed)
    
    # Round-trip accuracy
    @test maximum(abs.(percentage_values .- recovered)) < tolerance
    
    # Test with edge values
    edge_values = [0.0, 50.0, 100.0]
    edge_transformed = AutoGP.Transforms.apply(transform, edge_values)
    edge_recovered = AutoGP.Transforms.unapply(transform, edge_transformed)
    @test maximum(abs.(edge_values .- edge_recovered)) < tolerance
end

@testitem "get_autogp_transform PositiveTransform" setup=[AutoGPTransformData] begin
    transform = get_autogp_transform("positive", positive_values)
    
    @test transform isa PositiveTransform{Float64}
    @test transform.offset == 0.0  # No zeros in positive_values
    
    # Test AutoGP interface
    transformed = AutoGP.Transforms.apply(transform, positive_values)
    recovered = AutoGP.Transforms.unapply(transform, transformed)
    
    # Round-trip accuracy
    @test maximum(abs.(positive_values .- recovered)) < tolerance
end

@testitem "get_autogp_transform PositiveTransform with Zero" setup=[AutoGPTransformData] begin
    transform = get_autogp_transform("positive", zero_min_values)
    
    @test transform isa PositiveTransform{Float64}
    @test transform.offset > 0.0  # Should have offset for zero values
    @test transform.offset == 1.2 / 2  # Half the minimum positive value
    
    # Test AutoGP interface
    transformed = AutoGP.Transforms.apply(transform, zero_min_values)
    recovered = AutoGP.Transforms.unapply(transform, transformed)
    
    # Round-trip accuracy
    @test maximum(abs.(zero_min_values .- recovered)) < tolerance
end

@testitem "get_autogp_transform BoxCoxTransform" setup=[AutoGPTransformData] begin
    transform = get_autogp_transform("boxcox", boxcox_values)
    
    @test transform isa BoxCoxTransform{Float64, BoxCox.BoxCoxTransformation{Nothing}}
    @test transform.offset == 0.0  # No zeros in boxcox_values
    @test hasfield(typeof(transform), :λ)
    @test hasfield(typeof(transform), :max_values)
    
    # Test AutoGP interface
    transformed = AutoGP.Transforms.apply(transform, boxcox_values)
    recovered = AutoGP.Transforms.unapply(transform, transformed)
    
    # Round-trip accuracy
    @test maximum(abs.(boxcox_values .- recovered)) < tolerance
end

@testitem "AutoGP Transform Composition" setup=[AutoGPTransformData] begin
    # Create our custom transform
    pos_transform = get_autogp_transform("positive", positive_values)
    
    # Create AutoGP transforms
    linear_transform = AutoGP.Transforms.LinearTransform(2.0, 1.0)
    log_transform = AutoGP.Transforms.LogTransform()
    
    # Test composition
    composed = [linear_transform, pos_transform, log_transform]
    
    # Apply composed transform
    transformed = AutoGP.Transforms.apply(composed, positive_values)
    recovered = AutoGP.Transforms.unapply(composed, transformed)
    
    # Round-trip accuracy
    @test maximum(abs.(positive_values .- recovered)) < tolerance
end

@testitem "AutoGP Transform Single Element" setup=[AutoGPTransformData] begin
    # Test with single elements (scalar)
    single_value = [5.5]
    
    for transform_type in ["percentage", "positive", "boxcox"]
        if transform_type == "percentage"
            test_value = [50.0]  # Valid percentage
        else
            test_value = single_value
        end
        
        transform = get_autogp_transform(transform_type, test_value)
        transformed = AutoGP.Transforms.apply(transform, test_value)
        recovered = AutoGP.Transforms.unapply(transform, transformed)
        
        @test maximum(abs.(test_value .- recovered)) < tolerance
    end
end

@testitem "AutoGP Transform Error Conditions" setup=[AutoGPTransformData] begin
    # Test unknown transform name
    @test_throws AssertionError get_autogp_transform("unknown", positive_values)
    
    # Test empty values
    @test_throws AssertionError get_autogp_transform("positive", Float64[])
    
    # Test negative values (should fail in _get_offset)
    negative_values = [-1.0, 2.0, 3.0]
    @test_throws AssertionError get_autogp_transform("positive", negative_values)
end

@testitem "AutoGP vs Function-based Transform Consistency" setup=[AutoGPTransformData] begin
    # Compare AutoGP transforms with function-based transforms
    
    for transform_type in ["percentage", "positive", "boxcox"]
        local test_data
        if transform_type == "percentage"
            test_data = percentage_values
        else
            test_data = positive_values
        end
        
        # Get both versions
        autogp_transform = get_autogp_transform(transform_type, test_data)
        forward_func, inverse_func = get_transformations(transform_type, test_data)
        
        # Apply transformations
        autogp_transformed = AutoGP.Transforms.apply(autogp_transform, test_data)
        func_transformed = forward_func.(test_data)
        
        # Should produce same results
        @test maximum(abs.(autogp_transformed .- func_transformed)) < tolerance
        
        # Test inverse
        autogp_recovered = AutoGP.Transforms.unapply(autogp_transform, autogp_transformed)
        func_recovered = inverse_func.(func_transformed)
        
        @test maximum(abs.(autogp_recovered .- func_recovered)) < tolerance
    end
end

@testitem "AutoGP Transform Type Promotion" setup=[AutoGPTransformData] begin
    # Test with different numeric types
    int_values = [1, 2, 5, 10]
    
    transform = get_autogp_transform("positive", int_values)
    @test transform isa PositiveTransform{Int64}
    
    # Should still work with AutoGP interface
    transformed = AutoGP.Transforms.apply(transform, int_values)
    recovered = AutoGP.Transforms.unapply(transform, transformed)
    
    @test maximum(abs.(int_values .- recovered)) < tolerance
end

@testitem "autogp_inverse_transform Utility Function" setup=[AutoGPTransformData] begin
    # Test single transform
    transform = get_autogp_transform("positive", positive_values)
    inv_func = autogp_inverse_transform(transform)
    
    # Test the inverse function
    transformed = AutoGP.Transforms.apply(transform, positive_values)
    recovered_direct = AutoGP.Transforms.unapply(transform, transformed)
    recovered_func = inv_func.(transformed)
    
    @test maximum(abs.(recovered_direct .- recovered_func)) < tolerance
    @test maximum(abs.(positive_values .- recovered_func)) < tolerance
    
    # Test composed transforms
    linear_transform = AutoGP.Transforms.LinearTransform(2.0, 1.0)
    composed = [linear_transform, transform]
    inv_func_composed = autogp_inverse_transform(composed)
    
    composed_transformed = AutoGP.Transforms.apply(composed, positive_values)
    recovered_composed = inv_func_composed.(composed_transformed)
    
    @test maximum(abs.(positive_values .- recovered_composed)) < tolerance
end

@testitem "AutoGP Transform with Large Values" setup=[AutoGPTransformData] begin
    # Test with large values to check numerical stability
    large_values = [1e6, 1e7, 1e8]
    
    for transform_type in ["positive", "boxcox"]
        transform = get_autogp_transform(transform_type, large_values)
        transformed = AutoGP.Transforms.apply(transform, large_values)
        recovered = AutoGP.Transforms.unapply(transform, transformed)
        
        # Use relative tolerance for large values
        relative_error = maximum(abs.((large_values .- recovered) ./ large_values))
        @test relative_error < 1e-10
    end
end