"""
    make_and_fit_model(data; n_particles=8, smc_data_proportion=0.1, n_mcmc=200, n_hmc=50, kwargs...)

Create and fit a Gaussian Process (GP) model using Sequential Monte Carlo (SMC) sampling.

# Arguments
- `data`: A data structure containing the dataset (`data.ds`) and the target values (`data.y`).
- `n_particles`: The number of particles to use in the SMC sampling (default: 8).
- `smc_data_proportion`: The proportion of the data to use in each SMC step (default: 0.1).
- `n_mcmc`: The number of MCMC samples (default: 200).
- `n_hmc`: The number of HMC samples (default: 50).
- `kwargs...`: Additional keyword arguments to pass to the `AutoGP.fit_smc!` function.

# Returns
- `model`: The fitted GP model.
"""
function make_and_fit_model(
        data::TData; n_particles = 8, smc_data_proportion = 0.1,
        n_mcmc = 200, n_hmc = 50, kwargs...
)
    n_train = length(data.y)
    model = AutoGP.GPModel(data.ds, data.y; n_particles = n_particles)
    # Ensure smc_data_proportion results in at least step=1 for the schedule
    effective_proportion = max(smc_data_proportion, 1.0 / n_train)
    schedule = AutoGP.Schedule.linear_schedule(n_train, effective_proportion)
    AutoGP.fit_smc!(model; schedule = schedule, n_mcmc = n_mcmc, n_hmc = n_hmc, kwargs...)
    return model
end
