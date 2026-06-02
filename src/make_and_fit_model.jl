"""
    _stabilize_for_fit(y; flat_threshold=1.0e-3)

Guard against a degenerate (near-constant) transformed series before fitting.

Gaussian process fitting needs the data to have some spread: `AutoGP` rescales `y`
by its range, so a genuinely flat series makes the covariance singular
(`PosDefException`, issue #51). When the relative range of `y`
(`(max - min) / (|mean| + 1)`) falls below `flat_threshold`, add small Gaussian
jitter so the series is fittable; otherwise return `y` unchanged. Operates on the
(already transformed) values, so it is independent of which transformation was used.

This is a secondary safety net for data that is flat even after transformation. The
common case of a pathological Box-Cox λ on otherwise-fittable data is handled earlier,
in [`get_transformations`](@ref), by falling back to a log transformation.
"""
function _stabilize_for_fit(y::AbstractVector{<:Real}; flat_threshold = 1.0e-3)
    n = length(y)
    n > 1 || return y
    scale = abs(sum(y) / n) + 1
    rel_range = (maximum(y) - minimum(y)) / scale
    rel_range >= flat_threshold && return y  # enough spread → leave untouched
    σ = flat_threshold * scale
    @warn "Near-constant series (relative range $rel_range < $flat_threshold); adding \
jitter (σ = $σ) so the GP covariance stays positive-definite (issue #51)."
    return y .+ σ .* randn(n)
end

"""
    make_and_fit_model(data; n_particles=8, smc_data_proportion=0.1, n_mcmc=200, n_hmc=50, flat_threshold=1.0e-3, kwargs...)

Create and fit a Gaussian Process (GP) model using Sequential Monte Carlo (SMC) sampling.

# Arguments
- `data`: A data structure containing the dataset (`data.ds`) and the target values (`data.y`).
- `n_particles`: The number of particles to use in the SMC sampling (default: 8).
- `smc_data_proportion`: The proportion of the data to use in each SMC step (default: 0.1).
- `n_mcmc`: The number of MCMC samples (default: 200).
- `n_hmc`: The number of HMC samples (default: 50).
- `flat_threshold`: If the (transformed) target's relative range is below this, small
  Gaussian jitter is added so the GP covariance stays positive-definite (default: 1.0e-3).
  Series with enough spread are left untouched, preserving existing behaviour.
- `kwargs...`: Additional keyword arguments to pass to the `AutoGP.fit_smc!` function.

# Returns
- `model`: The fitted GP model.
"""
function make_and_fit_model(
        data::TData; n_particles = 8, smc_data_proportion = 0.1,
        n_mcmc = 200, n_hmc = 50, flat_threshold = 1.0e-3, kwargs...
    )
    n_train = length(data.y)
    y_fit = _stabilize_for_fit(data.y; flat_threshold = flat_threshold)
    model = AutoGP.GPModel(data.ds, y_fit; n_particles = n_particles)
    # Ensure smc_data_proportion results in at least step=1 for the schedule
    effective_proportion = max(smc_data_proportion, 1.0 / n_train)
    schedule = AutoGP.Schedule.linear_schedule(n_train, effective_proportion)
    AutoGP.fit_smc!(model; schedule = schedule, n_mcmc = n_mcmc, n_hmc = n_hmc, kwargs...)
    return model
end
