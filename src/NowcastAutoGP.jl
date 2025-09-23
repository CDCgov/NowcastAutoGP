module NowcastAutoGP
using AutoGP, Dates
using BoxCox: BoxCoxTransformation, confint, fit
using LogExpFunctions: logit, logistic

export TData
export create_transformed_data, get_transformations, get_autogp_transform, make_and_fit_model, forecast,
       forecast_with_nowcasts, create_nowcast_data
export PercentageTransform, PositiveTransform, BoxCoxTransform

include("transformations.jl")
include("TData.jl")
include("make_and_fit_model.jl")
include("create_nowcast_data.jl")
include("forecasting.jl")

end # module NowcastAutoGP
