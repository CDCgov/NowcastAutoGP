"""
    _inv_boxcox(λ::Real, offset::F, max_values) where {F}

Internal function to compute the inverse Box-Cox transformation with edge case handling.
"""
function _inv_boxcox(λ::Real, offset::F, max_values) where {F}
    function _inv(y)
        lambda_y_plus_1 = λ * y + one(F)

        # Handle edge cases based on λ sign and lambda_y_plus_1 value
        if λ > 0
            # Standard case: ensure lambda_y_plus_1 > 0
            safe_value = max(lambda_y_plus_1, F(1e-10))
            result = safe_value^(1/λ) - offset
        elseif λ < 0
            # Negative λ case: more careful handling
            if lambda_y_plus_1 > 1e-10
                # Normal case: lambda_y_plus_1 is sufficiently positive
                result = lambda_y_plus_1^(1/λ) - offset
            else
                # Edge case: lambda_y_plus_1 ≤ 0 or very small
                # For negative λ, (small positive)^(negative) = very large number
                # We'll clamp this to avoid numerical explosion
                if lambda_y_plus_1 <= 0
                    # Reduce to zero result since lambda_y_plus_1 == 0 ⟹ x == 0 when λ ≠ 0
                    # We collect this as probability mass at zero
                    result = zero(F)
                else
                    # lambda_y_plus_1 is very small but positive
                    clamped_result = lambda_y_plus_1^(1/λ)
                    # Clamp extremely large values to reasonable bounds
                    max_reasonable = F(1000 * max_values) # 1000x the max observed value
                    result = min(clamped_result, max_reasonable) - offset
                end
            end
        else
            # λ ≈ 0 case should have been handled above, but just in case
            result = exp(y) - offset
        end

        return max(result, zero(F))
    end
    return _inv
end

"""
    _get_offset(values::Vector{F}) where {F <: Real}

Internal function to compute an offset for transformations to ensure numerical stability.
"""
function _get_offset(values::Vector{F}) where {F <: Real}
    @assert !isempty(values) "Values array must not be empty"
    @assert all(values .>= zero(F)) "All values must be non-negative for the selected transformations"
    return minimum(values) == zero(F) ? minimum(values[values .> 0]) / 2 : zero(F)  # Half the minimum positive value for stability
end

"""
    get_transformations(transform_name::String, values::Vector{F}) where {F <: Real}

Return a tuple of transformation and inverse transformation functions for the specified transformation type.

This function creates appropriate data transformations for Gaussian Process modeling, where the goal is to
transform the input data to make it more suitable for modeling (typically more Gaussian-like) and then
provide the inverse transformation to convert predictions back to the original scale.

# Arguments
- `transform_name::String`: The name of the transformation to apply. Supported values:
  - `"percentage"`: For data bounded between 0 and 100 (e.g., percentages, rates)
  - `"positive"`: For strictly positive data (uses log transformation)
  - `"boxcox"`: Applies Box-Cox transformation with automatically fitted λ parameter
- `values::Vector{F}`: The input data values used to fit transformation parameters and determine offset

# Returns
A tuple `(forward_transform, inverse_transform)` where:
- `forward_transform`: Function that transforms data from original scale to transformed scale
- `inverse_transform`: Function that transforms data from transformed scale back to original scale

# Transformation Details

## Percentage Transformation
- **Use case**: Data bounded between 0 and 100 (percentages, rates)
- **Forward**: `y ↦ logit((y + offset) / 100)`
- **Inverse**: `y ↦ max(logistic(y) * 100 - offset, 0)`
- **Note**: Uses logit/logistic to map [0,100] to (-∞,∞) and back

## Positive Transformation
- **Use case**: Strictly positive continuous data
- **Forward**: `y ↦ log(y + offset)`
- **Inverse**: `y ↦ max(exp(y) - offset, 0)`
- **Note**: Log transformation for positive data with offset for numerical stability

## Box-Cox Transformation
- **Use case**: General purpose transformation for positive data
- **Forward**: `y ↦ BoxCox_λ(y + offset)` where λ is automatically fitted
- **Inverse**: Custom inverse function handling edge cases for numerical stability
- **Note**: Automatically determines optimal λ parameter via maximum likelihood

# Offset Calculation
An offset is automatically calculated using `_get_offet(values)`:
- If minimum value is 0: offset = (minimum positive value) / 2
- Otherwise: offset = 0
- Purpose: Ensures numerical stability and handles boundary cases

# Examples
```julia
# Percentage data (0-100 range)
values = [10.5, 25.3, 67.8, 89.2]
forward, inverse = get_transformations("percentage", values)
transformed = forward.(values)
recovered = inverse.(transformed)

# Strictly positive data
values = [1.2, 3.4, 8.9, 15.6]
forward, inverse = get_transformations("positive", values)

# General positive data with automatic Box-Cox fitting
values = [0.1, 0.5, 2.3, 5.7, 12.1]
forward, inverse = get_transformations("boxcox", values)
```

# Throws
- `AssertionError`: If `transform_name` is not one of the supported transformation types
- `AssertionError`: Via `_get_offet` if `values` is empty or contains negative values

# See Also
- [`_get_offset`](@ref): Calculates the offset value for numerical stability
- [`_inv_boxcox`](@ref): Handles inverse Box-Cox transformation with edge case handling
"""
function get_transformations(
        transform_name::String, values::Vector{F}) where {F <: Real}
    offset = _get_offset(values)
    if transform_name == "percentage"
        @info "Using percentage transformation"
        return (y -> logit((y + offset) / 100), y -> max(logistic(y) * 100 - offset, zero(F)))
    elseif transform_name == "positive"
        @info "Using positive transformation with offset = $offset"
        return (y -> log(y + offset), y -> max(exp(y) - offset, zero(F)))
    elseif transform_name == "boxcox"
        max_values = maximum(values)
        bc = fit(BoxCoxTransformation, values .+ offset) # Fit Box-Cox transformation
        λ = bc.λ
        @info "Using Box-Cox transformation with λ = $λ and offset = $offset"
        return (y -> bc(y + offset), _inv_boxcox(λ, offset, max_values))
    else
        throw(AssertionError("Unknown transform_name: $transform_name"))
    end
end
