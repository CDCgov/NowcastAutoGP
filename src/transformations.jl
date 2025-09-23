# AutoGP-compatible transform structs
"""
    PercentageTransform{F} <: AutoGP.Transforms.Transform

An AutoGP-compatible transform for percentage data (0-100 range) using logit transformation.

Uses logit((y + offset) / 100) for forward transform and logistic(y) * 100 - offset for inverse.
"""
struct PercentageTransform{F <: Real} <: AutoGP.Transforms.Transform
    offset::F
end

"""
    PositiveTransform{F} <: AutoGP.Transforms.Transform

An AutoGP-compatible transform for positive data using log transformation with offset.

Uses log(y + offset) for forward transform and exp(y) - offset for inverse.
"""
struct PositiveTransform{F <: Real} <: AutoGP.Transforms.Transform
    offset::F
end

"""
    BoxCoxTransform{F, B} <: AutoGP.Transforms.Transform

An AutoGP-compatible transform for positive data using Box-Cox transformation.

Uses fitted Box-Cox transformation with automatic λ parameter and offset handling.
"""
struct BoxCoxTransform{F <: Real, B} <: AutoGP.Transforms.Transform
    offset::F
    boxcox::B
    λ::F
    max_values::F
end

# Implement AutoGP.Transforms interface for our custom transforms
function AutoGP.Transforms.apply(t::PercentageTransform, x)
    return logit.((x .+ t.offset) ./ 100)
end

function AutoGP.Transforms.unapply(t::PercentageTransform{F}, y) where {F}
    return max.(logistic.(y) .* 100 .- t.offset, zero(F))
end

function AutoGP.Transforms.apply(t::PositiveTransform, x)
    return log.(x .+ t.offset)
end

function AutoGP.Transforms.unapply(t::PositiveTransform{F}, y) where {F}
    return max.(exp.(y) .- t.offset, zero(F))
end

function AutoGP.Transforms.apply(t::BoxCoxTransform, x)
    return t.boxcox.(x .+ t.offset)
end

function AutoGP.Transforms.unapply(t::BoxCoxTransform{F}, y) where {F}
    inv_func = _inv_boxcox(t.λ, t.offset, t.max_values)
    return inv_func.(y)
end

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
    get_autogp_transform(transform_name::String, values::Vector{F}) where {F <: Real}

Create an AutoGP-compatible transform for the specified transformation type.

This function creates appropriate data transformations compatible with AutoGP's transform system,
which provides better integration with AutoGP's prediction pipeline and supports transform composition.

# Arguments
- `transform_name::String`: The name of the transformation to apply. Supported values:
  - `"percentage"`: For data bounded between 0 and 100 (e.g., percentages, rates)  
  - `"positive"`: For strictly positive data (uses log transformation)
  - `"boxcox"`: Applies Box-Cox transformation with automatically fitted λ parameter
- `values::Vector{F}`: The input data values used to fit transformation parameters and determine offset

# Returns
An AutoGP.Transforms.Transform object that can be used with AutoGP's apply/unapply functions
and supports composition with other transforms.

# Examples
```julia
# Create AutoGP transform for percentage data
values = [10.5, 25.3, 67.8, 89.2]
transform = get_autogp_transform("percentage", values)
transformed = AutoGP.Transforms.apply(transform, values)
recovered = AutoGP.Transforms.unapply(transform, transformed)

# Compose with other AutoGP transforms
log_transform = AutoGP.Transforms.LogTransform()
combined = [transform, log_transform]
```

# See Also
- [`get_transformations`](@ref): Original function-based interface (maintained for compatibility)
- [`PercentageTransform`](@ref), [`PositiveTransform`](@ref), [`BoxCoxTransform`](@ref): Transform implementations
- [`autogp_inverse_transform`](@ref): Extract inverse transform function for use with forecast functions
"""
function get_autogp_transform(
        transform_name::String, values::Vector{F}) where {F <: Real}
    offset = _get_offset(values)
    if transform_name == "percentage"
        @info "Using AutoGP percentage transformation with offset = $offset"
        return PercentageTransform(offset)
    elseif transform_name == "positive"
        @info "Using AutoGP positive transformation with offset = $offset"
        return PositiveTransform(offset)
    elseif transform_name == "boxcox"
        max_values = maximum(values)
        bc = fit(BoxCoxTransformation, values .+ offset)
        λ = bc.λ
        @info "Using AutoGP Box-Cox transformation with λ = $λ and offset = $offset"
        return BoxCoxTransform(offset, bc, λ, max_values)
    else
        throw(AssertionError("Unknown transform_name: $transform_name"))
    end
end

"""
    autogp_inverse_transform(transform::AutoGP.Transforms.Transform)

Extract an inverse transformation function from an AutoGP transform for use with forecasting functions.

This utility function allows AutoGP transforms to be used with the existing forecasting interface
that expects inverse transformation functions.

# Arguments
- `transform`: An AutoGP.Transforms.Transform object

# Returns
A function that can be broadcast over forecast arrays to apply the inverse transformation.

# Examples
```julia
# Create AutoGP transform and use with forecasting
transform = get_autogp_transform("positive", values)
data = TData(dates, values; transformation = x -> AutoGP.Transforms.apply(transform, x))
model = make_and_fit_model(data)

# Use with forecast function
inv_func = autogp_inverse_transform(transform)
forecasts = forecast(model, forecast_dates, 100; inv_transformation = inv_func)
```

# See Also
- [`forecast`](@ref): Forecast function that accepts inv_transformation parameter
- [`forecast_with_nowcasts`](@ref): Nowcast-aware forecasting with inverse transformation
"""
function autogp_inverse_transform(transform::AutoGP.Transforms.Transform)
    return y -> AutoGP.Transforms.unapply(transform, y)
end

"""
    autogp_inverse_transform(transforms::Vector{<:AutoGP.Transforms.Transform})

Extract an inverse transformation function from a composed AutoGP transform chain.

# Arguments
- `transforms`: A vector of AutoGP.Transforms.Transform objects representing a composition

# Returns
A function that applies the inverse of the composed transforms in reverse order.
"""
function autogp_inverse_transform(transforms::Vector{<:AutoGP.Transforms.Transform})
    return y -> AutoGP.Transforms.unapply(transforms, y)
end

"""
    get_transformations(transform_name::String, values::Vector{F}) where {F <: Real}

Return a tuple of transformation and inverse transformation functions for the specified transformation type.

This function creates appropriate data transformations for Gaussian Process modeling, where the goal is to
transform the input data to make it more suitable for modeling (typically more Gaussian-like) and then
provide the inverse transformation to convert predictions back to the original scale.

**Note**: This function maintains the original function-based interface for backward compatibility.
For better integration with AutoGP, consider using [`get_autogp_transform`](@ref).

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
- [`get_autogp_transform`](@ref): AutoGP-compatible transform interface (recommended)
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
