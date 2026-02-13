"""
    create_nowcast_data(nowcasts::AbstractVector, dates::Vector{Date}; transformation = y -> y)

Create nowcast data structures from a vector of nowcast scenarios.

# Arguments
- `nowcasts`: A vector where each element is a vector of nowcast values representing one scenario.
  All inner vectors must have the same length as `dates`.
- `dates`: A vector of Date objects corresponding to the nowcast time points.
- `transformation`: A function to apply to the nowcast values (default: identity).

# Returns
A vector of NamedTuples, where each NamedTuple represents one nowcast scenario with fields:
- `ds`: The dates vector
- `y`: The transformed nowcast values
- `values`: The original (untransformed) nowcast values

# Example
```julia
# Two nowcast scenarios for 3 dates
nowcasts = [[10.5, 11.2, 12.1], [9.8, 10.9, 11.5]]
dates = [Date(2024,1,1), Date(2024,1,2), Date(2024,1,3)]
nowcast_data = create_nowcast_data(nowcasts, dates; transformation = log)
# Returns vector of 2 NamedTuples, each with transformed and original values
```
"""
function create_nowcast_data(
        nowcasts::AbstractVector, dates::Vector{Date}; transformation = y -> y
    )
    @assert all(length.(nowcasts) .== length(dates)) "Length of each nowcast must match length of dates"
    @assert !isempty(nowcasts) "nowcasts must not be empty"
    # Check all vectors have the same length
    first_length = length(nowcasts[1])
    @assert all(length(v) == first_length for v in nowcasts) "All vectors in nowcasts must have the same length"

    nowcast_data = map(nowcasts) do nowcast
        create_transformed_data(dates, nowcast; transformation)
    end
    return nowcast_data
end

"""
    create_nowcast_data(nowcasts::AbstractMatrix, dates::Vector{Date}; transformation = y -> y)

Create nowcast data structures from a matrix of nowcast scenarios.

# Arguments
- `nowcasts`: A matrix where each column represents one nowcast scenario. The number of rows
  must match the length of `dates`.
- `dates`: A vector of Date objects corresponding to the nowcast time points.
- `transformation`: A function to apply to the nowcast values (default: identity).

# Returns
A vector of NamedTuples, where each NamedTuple represents one nowcast scenario with fields:
- `ds`: The dates vector
- `y`: The transformed nowcast values
- `values`: The original (untransformed) nowcast values

# Notes
This method converts the matrix to a vector of columns internally and delegates to the vector method.

# Example
```julia
# Matrix with 3 time points (rows) and 2 scenarios (columns)
nowcasts = [10.5 9.8; 11.2 10.9; 12.1 11.5]
dates = [Date(2024,1,1), Date(2024,1,2), Date(2024,1,3)]
nowcast_data = create_nowcast_data(nowcasts, dates; transformation = log)
# Returns vector of 2 NamedTuples, each with transformed and original values
```
"""
function create_nowcast_data(
        nowcasts::AbstractMatrix, dates::Vector{Date}; transformation = y -> y
    )
    _nowcasts = eachcol(nowcasts) |> collect
    return create_nowcast_data(_nowcasts, dates; transformation)
end
