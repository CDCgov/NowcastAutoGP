
"""
    TData{D, F}

A container for transformed time series data used in nowcasting models.

# Type Parameters
- `D`: Type for dates/timestamps (e.g., `Date`, `DateTime`)
- `F`: Type for numeric values, automatically promoted from input types

# Fields
- `ds::Vector{D}`: Vector of dates or timestamps corresponding to observations
- `y::Vector{F}`: Vector of transformed target values (result of applying transformation)
- `values::Vector{F}`: Vector of original values, converted to common type with `y`

# Constructor
    TData(ds, values; transformation)

Create a `TData` instance by applying a transformation to the input values.

## Arguments
- `ds`: Vector of dates or timestamps
- `values`: Vector of original numeric values
- `transformation`: Function to apply element-wise to `values` to create `y`

The constructor automatically promotes types using `promote_type` to ensure
`y` and `values` have compatible numeric types.

# Example
```julia
using Dates

dates = [Date(2023, 1, 1), Date(2023, 1, 2), Date(2023, 1, 3)]
raw_values = [10, 20, 30]

# Apply log transformation
tdata = TData(dates, raw_values; transformation = log)

# Apply custom transformation
tdata = TData(dates, raw_values; transformation = x -> (x - mean(raw_values)) / std(raw_values))
```

# Validation
The constructor ensures that `ds` and `values` have the same length and throws an
`ArgumentError` if they don't match.
"""
struct TData{D, F}
    ds::Vector{D}
    y::Vector{F}
    values::Vector{F}

    function TData(ds::Vector{D}, values::Vector{V}; transformation) where {D, V}
        @assert length(ds) == length(values) "length of `ds` should match length of `values`"

        # Apply the transformation to the target values
        y = transformation.(values)

        # Find common type and convert both vectors
        F = promote_type(eltype(y), V)
        converted_y = convert.(F, y)
        converted_values = convert.(F, values)

        new{D, F}(ds, converted_y, converted_values)
    end
end

# Convenience constructor that infers types from any iterable inputs
"""
    create_transformed_data(ds, values; transformation)

Convenience function to create a `TData` instance from any iterable inputs of dates/times and values.
"""
function create_transformed_data(ds, values; transformation)
    TData(collect(ds), collect(values); transformation = transformation)
end
