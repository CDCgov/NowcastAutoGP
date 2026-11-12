"""
    PoissonObservation()

Observation model for count data: counts are Poisson distributed given a latent
log-intensity, and the log-intensity is the Gaussian process plus white noise,

```math
y_t \\sim \\mathrm{Poisson}(\\exp(\\eta_t)), \\qquad
\\eta_t = c_0 + c_1 \\, (f_t + \\varepsilon_t), \\qquad
f \\sim \\mathcal{GP}(0, k), \\qquad \\varepsilon_t \\sim \\mathcal{N}(0, \\sigma^2).
```

The constants ``c_0`` and ``c_1`` are fixed from the training data so that ``f`` is on the
unit scale that the `AutoGP` priors assume.

Pass it to [`create_count_data`](@ref). Counts are used as they are: no transformation and
no offset for zeros.

# Inference

The latent log-intensities are not sampled. Each count is replaced by a Gaussian
pseudo-observation of ``\\eta_t`` with mean ``\\psi(y_t + 1/2)`` and variance
``\\psi'(y_t + 1/2)`` (digamma and trigamma; approximately ``\\log y_t`` and ``1 / y_t``),
which matches the first two moments of the Poisson likelihood as a function of ``\\eta_t``.
The model is then a Gaussian process regression with a known noise variance for each data
point, so fitting costs the same as for a transformed series. Forecasts draw the latent
log-intensity and then draw Poisson counts.

The approximation is accurate for counts of about 10 or more and weakest for runs of zeros,
where it keeps the fitted rate above roughly 0.14 per time step.
"""
struct PoissonObservation end

"""
    CountData{D, O}

A container for count time series data to be modelled with a count observation model.

# Fields
- `ds::Vector{D}`: Dates or timestamps of the observations.
- `values::Vector{Float64}`: Observed counts. Non-integer values are allowed, since nowcast
  estimates need not be integers.
- `observation_model::O`: The observation model, e.g. [`PoissonObservation`](@ref).

# Constructor
    CountData(ds, values; observation_model = PoissonObservation())

Throws an `ArgumentError` if `ds` and `values` differ in length, or if any value is
negative or not finite.
"""
struct CountData{D, O}
    ds::Vector{D}
    values::Vector{Float64}
    observation_model::O

    function CountData(
            ds::Vector{D}, values::AbstractVector{<:Real};
            observation_model::O = PoissonObservation()
        ) where {D, O}
        length(ds) == length(values) ||
            throw(ArgumentError("length of `ds` should match length of `values`"))
        _valid_counts(values) ||
            throw(ArgumentError("count `values` must be finite and non-negative"))
        return new{D, O}(ds, Float64.(values), observation_model)
    end
end

_valid_counts(values) = all(v -> isfinite(v) && v >= 0, values)

"""
    create_count_data(ds, values; observation_model = PoissonObservation())

Create a [`CountData`](@ref) instance from any iterable inputs of dates/times and counts.
This is the count-data counterpart of [`create_transformed_data`](@ref): the result is
passed to [`make_and_fit_model`](@ref), and forecasts from the fitted model are counts.

# Example
```julia
using Dates

dates = Date(2024, 1, 6):Week(1):Date(2024, 6, 29)
counts = rand(0:8, length(dates))
data = create_count_data(dates, counts)
model = make_and_fit_model(data; n_particles = 8, n_mcmc = 50, n_hmc = 50)
forecasts = forecast(model, dates[end] .+ Week.(1:4), 200)   # Matrix of counts
```
"""
function create_count_data(ds, values; observation_model = PoissonObservation())
    return CountData(collect(ds), collect(values); observation_model = observation_model)
end

# Offset in the pseudo-observations, so that a count of zero is defined.
const SITE_OFFSET = 0.5
# Largest Poisson intensity used when sampling forecasts.
const MAX_INTENSITY = 1.0e15

# Mean and variance of the Gaussian pseudo-observation of the log-intensity for a count.
_site_mean(y::Real) = digamma(y + SITE_OFFSET)
_site_var(y::Real) = trigamma(y + SITE_OFFSET)

"""
    PoissonFixedSites(c0, c1)
    PoissonFixedSites(values)

`Gen` distribution used as the `AutoGP` observation model for [`PoissonObservation`](@ref).
It scores a vector of counts `y` given the noise-free kernel matrix `K` and the white noise
variance `noise` as

    N(z(y); 0, K + noise * I + Diagonal(v(y)))

where `z` and `v` are the pseudo-observations and their variances on the scale of the
Gaussian process, i.e. after subtracting `c0` and dividing by `c1`. The density is defined
up to a factor that depends only on `y`, which is the same for every kernel and so does not
affect inference.

The second constructor sets `c0` to the mean and `c1` to the range of the
pseudo-observations of `values`, the same centering and scaling `AutoGP` applies to
continuous data. For a series that is flat within its count noise, `c1` is instead the
typical standard deviation of the pseudo-observations.
"""
struct PoissonFixedSites <: Gen.Distribution{Vector{Float64}}
    c0::Float64
    c1::Float64
end

function PoissonFixedSites(values::AbstractVector{<:Real})
    m = _site_mean.(values)
    v = _site_var.(values)
    c0 = sum(m) / length(m)
    c1 = max(maximum(m) - minimum(m), sqrt(sum(v) / length(v)))
    return PoissonFixedSites(c0, c1)
end

# Pseudo-observations and their variances on the scale of the Gaussian process.
function _sites(d::PoissonFixedSites, y::AbstractVector{<:Real})
    z = (_site_mean.(y) .- d.c0) ./ d.c1
    v = _site_var.(y) ./ d.c1^2
    return z, v
end

function Gen.logpdf(
        d::PoissonFixedSites, y::AbstractVector{<:Real}, K::AbstractMatrix{<:Real},
        noise::Real
    )
    isempty(y) && return 0.0
    _valid_counts(y) || return -Inf
    z, v = _sites(d, y)
    return Gen.logpdf(Gen.mvnormal, z, zeros(length(z)), K + noise * I + Diagonal(v))
end

function Gen.logpdf_grad(
        d::PoissonFixedSites, y::AbstractVector{<:Real}, K::AbstractMatrix{<:Real},
        noise::Real
    )
    isempty(y) && return (nothing, zeros(0, 0), 0.0)
    z, v = _sites(d, y)
    (_, _, dK) = Gen.logpdf_grad(
        Gen.mvnormal, z, zeros(length(z)), K + noise * I + Diagonal(v)
    )
    return (nothing, dK, tr(dK))
end

function Gen.random(d::PoissonFixedSites, K::AbstractMatrix{<:Real}, noise::Real)
    n = size(K, 1)
    n == 0 && return Float64[]
    g = Gen.random(Gen.mvnormal, zeros(n), K + noise * I)
    return Float64.(_sample_counts(d.c0 .+ d.c1 .* g))
end

(d::PoissonFixedSites)(K, noise) = Gen.random(d, K, noise)
Gen.has_output_grad(::PoissonFixedSites) = false
Gen.has_argument_grads(::PoissonFixedSites) = (true, true)

# Poisson counts given log-intensities.
_sample_counts(eta) = rand.(Poisson.(min.(exp.(eta), MAX_INTENSITY)))

# The `AutoGP` observation distribution for an observation model and its training data.
_observation_distribution(::PoissonObservation, values) = PoissonFixedSites(values)

"""
    _site_conditional(kernel, noise, ts, z, v, ts_pred)

Mean and covariance of the Gaussian process plus white noise at `ts_pred`, given
pseudo-observations `z` at `ts` with noise variances `noise .+ v`. With `v` all zero this is
the standard Gaussian process predictive that `AutoGP` uses for Gaussian observations.
"""
function _site_conditional(
        kernel, noise::Real, ts::Vector{Float64}, z::Vector{Float64},
        v::Vector{Float64}, ts_pred::Vector{Float64}
    )
    n = length(ts)
    K = eval_cov(kernel, vcat(ts, ts_pred))
    K11 = K[1:n, 1:n] + noise * I + Diagonal(v)
    K12 = K[1:n, (n + 1):end]
    K22 = K[(n + 1):end, (n + 1):end]
    F = cholesky(Symmetric(K11))
    mu = K12' * (F \ z)
    Sigma = K22 - K12' * (F \ K12)
    Sigma = (Sigma + Sigma') / 2 + noise * I
    return mu, Sigma
end

# Mixture over particles of the predictive distribution of the log-intensity at `dates`.
function _log_intensity_predictive(model::AutoGP.GPModel, d::PoissonFixedSites, dates)
    ts = Float64.(AutoGP.Transforms.apply(model.ds_transform, AutoGP.to_numeric.(model.ds)))
    ts_pred = Float64.(AutoGP.Transforms.apply(model.ds_transform, AutoGP.to_numeric.(dates)))
    z, v = _sites(d, model.y)
    kernels = AutoGP.covariance_kernels(model; reparameterize = false)
    noises = AutoGP.observation_noise_variances(model; reparameterize = false)
    components = map(kernels, noises) do kernel, noise
        mu, Sigma = _site_conditional(kernel, noise, ts, z, v, ts_pred)
        MvNormal(d.c0 .+ d.c1 .* mu, Symmetric(d.c1^2 .* Sigma))
    end
    return MixtureModel(components, AutoGP.particle_weights(model))
end

# Draws from the predictive distribution of a fitted model, on the scale the model was
# fitted on. Returns a matrix with one row per date and one column per draw.
function _sample_predictive(model::AutoGP.GPModel, dates, n_draws::Int)
    return _sample_predictive(model.config.observation_model, model, dates, n_draws)
end

function _sample_predictive(::Nothing, model::AutoGP.GPModel, dates, n_draws::Int)
    return rand(AutoGP.predict_mvn(model, dates), n_draws)
end

function _sample_predictive(
        d::PoissonFixedSites, model::AutoGP.GPModel, dates, n_draws::Int
    )
    eta = rand(_log_intensity_predictive(model, d, dates), n_draws)
    return _sample_counts(eta)
end

# Check that data added to a fitted model is valid for its observation model.
_check_new_observations(::Nothing, y) = nothing
function _check_new_observations(::PoissonFixedSites, y)
    _valid_counts(y) || throw(
        ArgumentError(
            "the model has a count observation model, so nowcast values must be finite " *
                "and non-negative counts; create them with `create_nowcast_data` without " *
                "a transformation"
        )
    )
    return nothing
end
