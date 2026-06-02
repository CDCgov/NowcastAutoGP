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
function _stabilize_for_fit(y::AbstractVector{<:Real}; flat_threshold)
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
    function make_and_fit_model(
        data::TData; n_particles, smc_data_proportion,
        flat_threshold = 1.0e-3, config = AutoGP.GP.GPConfig(), kwargs...
    )

Create and fit a Gaussian Process (GP) model using Sequential Monte Carlo (SMC) sampling.

# Arguments
- `data`: A data structure containing the dataset (`data.ds`) and the target values (`data.y`).
- `n_particles`: The number of particles to use in the SMC sampling.
- `smc_data_proportion`: The proportion of the data to use in each SMC step.
- `flat_threshold`: If the (transformed) target's relative range is below this, small
  Gaussian jitter is added so the GP covariance stays positive-definite (default: 1.0e-3).
  Series with enough spread are left untouched, preserving existing behaviour.
- `config`: An `AutoGP.GP.GPConfig` describing the GP prior — kernel-structure distribution
  (`node_dist_leaf`/`node_dist_nocp`/`node_dist_cp`) and hyperparameter priors
  (`prior[:gamma]`/`prior[:period]`/`prior[:wildcard]`). Defaults to `GPConfig()` (AutoGP defaults),
  so existing behaviour is unchanged. Build a customised one with the `GPConfig` keyword
  constructor — see Examples.
- `kwargs...`: Additional keyword arguments forwarded to `AutoGP.fit_smc!`. Note that `fit_smc!`
  *requires* `n_mcmc` and `n_hmc` (number of MCMC structure steps and HMC parameter steps per SMC
  step), so these must be supplied here; other `fit_smc!` options (e.g. `hmc_config`, `biased`,
  `shuffle`, `verbose`) may also be passed.

# Returns
- `model`: The fitted GP model.

# Examples
```julia
# Default prior:
model = make_and_fit_model(data; n_mcmc = 200, n_hmc = 50)

# Custom prior — pass a GPConfig. Kernel-structure fields go straight to the constructor,
# e.g. give SquaredExponential (code 3) leaf-kernel prior mass:
config = GPConfig(node_dist_leaf = [0.0, 0.25, 0.25, 0.25, 0.25])
model = make_and_fit_model(data; config = config, n_mcmc = 200, n_hmc = 50)

# To tweak a single nested prior entry (e.g. re-centre the periodic component near an annual
# cycle) copy the default prior and update the leaf:
prior = deepcopy(GPConfig().prior)
prior[:period][:mu] = log(1.0)
model = make_and_fit_model(data; config = GPConfig(prior = prior), n_mcmc = 200, n_hmc = 50)
```

`@set` from [Accessors.jl](https://github.com/JuliaObjects/Accessors.jl) is a handy alternative
for these copy-and-update edits if you have it in your environment
(`config = @set GPConfig().prior[:period][:mu] = log(1.0)`); it is not a dependency of this package.
"""
function make_and_fit_model(
        data::TData; n_particles, smc_data_proportion,
        flat_threshold = 1.0e-3, config = AutoGP.GP.GPConfig(), kwargs...
    )
    n_train = length(data.y)
    y_fit = _stabilize_for_fit(data.y; flat_threshold = flat_threshold)
    model = AutoGP.GPModel(
        data.ds, y_fit; n_particles = n_particles,
        config = config
    )
    # Ensure smc_data_proportion results in at least step=1 for the schedule
    effective_proportion = max(smc_data_proportion, 1.0 / n_train)
    schedule = AutoGP.Schedule.linear_schedule(n_train, effective_proportion)
    AutoGP.fit_smc!(model; schedule = schedule, kwargs...)
    return model
end
