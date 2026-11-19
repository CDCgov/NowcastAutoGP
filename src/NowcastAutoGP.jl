module NowcastAutoGP
using AutoGP, Dates
using BoxCox: BoxCoxTransformation, confint, fit
using Distributions: MixtureModel, MvNormal, Poisson
using LinearAlgebra: BLAS, Diagonal, I, Symmetric, cholesky, tr
using LogExpFunctions: logit, logistic
using Printf: @sprintf
using ProgressMeter: @showprogress, Progress, next!
using SpecialFunctions: digamma, trigamma

import AutoGP.GP: LeafNode, eval_cov, reparameterize, rescale, pretty, LinearTransform
import Gen

const GPModel = AutoGP.GPModel # re-exporting for convenience
const GPConfig = AutoGP.GP.GPConfig # re-exporting for convenience
export TData, GPModel, GPConfig, RandomWalk, IntegratedBrownianMotion
export CountData, PoissonObservation
export create_transformed_data, get_transformations, make_and_fit_model, forecast,
    forecast_with_nowcasts, create_nowcast_data, create_count_data

include("transformations.jl")
include("TData.jl")
include("extension_kernels.jl")
include("count_observation.jl")
include("make_and_fit_model.jl")
include("create_nowcast_data.jl")
include("forecasting.jl")

end # module NowcastAutoGP
