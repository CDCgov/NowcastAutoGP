module NowcastAutoGP
using AutoGP, Dates
using BoxCox: BoxCoxTransformation, confint, fit
using LinearAlgebra: BLAS
using LogExpFunctions: logit, logistic
using ProgressMeter: @showprogress, Progress, next!

const GPModel = AutoGP.GPModel # re-exporting for convenience
export TData, GPModel
export create_transformed_data, get_transformations, make_and_fit_model, forecast,
    forecast_with_nowcasts, create_nowcast_data

include("transformations.jl")
include("TData.jl")
include("make_and_fit_model.jl")
include("create_nowcast_data.jl")
include("forecasting.jl")

end # module NowcastAutoGP
