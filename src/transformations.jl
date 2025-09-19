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

function get_transformations(
        transform_name::String, values::Vector{F}) where {F <: Real}
    offset = minimum(values) == zero(F) ? minimum(values[values .> 0]) / 2 : zero(F)  # Half the minimum positive value for stability
    if transform_name == "percentage"
        @info "Using percentage transformation"
        return (y -> logit(y / 100), y -> logistic(y) * 100)
    elseif transform_name == "positive"
        @info "Using positive transformation with offset = $offset"
        return (y -> log(y + offset), y -> max(exp(y) - offset, 0.0))
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
