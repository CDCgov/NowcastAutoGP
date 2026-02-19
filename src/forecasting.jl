"""Temporarily set BLAS threads to 1 to avoid deadlock with AutoGP's internal `Threads.@threads`."""
function _with_single_blas(f)
    n = BLAS.get_num_threads()
    BLAS.set_num_threads(1)
    try
        return f()
    finally
        BLAS.set_num_threads(n)
    end
end

"""
    forecast(model, forecast_dates, forecast_draws::Int)

Generate forecast samples from a fitted `AutoGP` model.

# Arguments
- `model`: Fitted `AutoGP.GPModel`.
- `forecast_dates`: Vector or range of dates to predict.
- `forecast_draws`: Number of samples to draw.

# Keyword arguments
- `inv_transformation`: Function applied elementwise to map forecasts back to the original scale (default: identity).
- `forecast_n_hmc`: If `nothing`, draw from the current model state. If an `Int`, run that many HMC parameter steps before each draw (default: `nothing`).
- `verbose`: If `true`, display progress information during forecasting (default: `false`).

# Returns
- A matrix of samples with size `(length(forecast_dates), forecast_draws)`.
"""
function forecast(
        model::AutoGP.GPModel, forecast_dates, forecast_draws::Int;
        inv_transformation = y -> y, forecast_n_hmc::Union{Int, Nothing} = nothing
    )
    return _forecast(
        model, forecast_dates, forecast_draws, forecast_n_hmc;
        inv_transformation = inv_transformation
    )
end

function _forecast(
        model, forecast_dates, forecast_draws::Int, ::Nothing;
        inv_transformation = y -> y
    )
    # Convert forecast_dates to vector if it's a range
    dates_vector = collect(forecast_dates)
    dist = AutoGP.predict_mvn(model, dates_vector)
    _forecasts = rand(dist, forecast_draws)
    # Apply inverse transformation to the forecasts
    forecasts = inv_transformation.(_forecasts)
    return forecasts
end

function _forecast(
        model, forecast_dates, forecast_draws::Int, forecast_n_hmc::Int;
        inv_transformation = y -> y
    )
    # Convert forecast_dates to vector if it's a range
    dates_vector = collect(forecast_dates)
    n_dates = length(dates_vector)
    _forecasts = _with_single_blas() do
        out = Matrix{Float64}(undef, n_dates, forecast_draws)
        for i in 1:forecast_draws
            # Refine the GP models with HMC steps for each forecast draw
            AutoGP.mcmc_parameters!(model, forecast_n_hmc)
            dist = AutoGP.predict_mvn(model, dates_vector)
            out[:, i] = rand(dist)
        end
        return out
    end

    # Apply inverse transformation to the forecasts
    forecasts = inv_transformation.(_forecasts)
    return forecasts
end

"""
    forecast_with_nowcasts(base_model, nowcasts, forecast_dates, forecast_draws_per_nowcast;
                          inv_transformation = y -> y, n_mcmc = 0, n_hmc = 0, ess_threshold = 0.0)

Generate forecasts by conditioning on multiple nowcast scenarios.

# Arguments
- `base_model`: Fitted `AutoGP.GPModel` trained on confirmed (non-nowcast) data.
- `nowcasts`: Vector of `TData` scenarios with fields `ds`, `y`, and `values`.
- `forecast_dates`: Vector or range of dates to predict.
- `forecast_draws_per_nowcast`: Samples per scenario.

# Keyword arguments
- `inv_transformation`: Function applied elementwise to map forecasts back to the original scale (default: identity).
- `n_mcmc`: Number of MCMC structure steps after adding each nowcast (default: 0). If `> 0`, `n_hmc` must also be `> 0`.
- `n_hmc`: Number of HMC parameter steps per MCMC step (default: 0). Can be `> 0` even if `n_mcmc == 0`.
- `ess_threshold`: Effective sample size threshold for particle resampling, as a fraction of total particles (default: 0.0).
- `forecast_n_hmc`: Number of HMC steps to run before each forecast draw (default: `nothing`). If `nothing`, no HMC steps are taken during forecasting.
- `verbose`: If `true`, display progress information during forecasting (default: `false`).

# Returns
- A matrix with size `(length(forecast_dates), length(nowcasts) * forecast_draws_per_nowcast)`.

# Notes
- Each scenario operates on an independent copy of the base model, so the original model is never mutated.
- `n_mcmc == 0 && n_hmc > 0` performs parameter-only updates to the _particle ensemble_; `n_mcmc > 0 && n_hmc > 0` performs full MCMC.
- `forecast_n_hmc` is independent of `n_mcmc` and `n_hmc` and controls HMC steps only during forecasting, not during nowcast incorporation.
If `n_mcmc == 0 && n_hmc == 0 && forecast_n_hmc > 0`, HMC steps are only taken during forecasting, not during nowcast incorporation.


# Example
```julia
nowcast_scenarios = [
    (ds = [Date(2024,1,1), Date(2024,1,2)], y = [10.5, 11.2], values = [10.5, 11.2]),
    (ds = [Date(2024,1,1), Date(2024,1,2)], y = [9.8, 10.9], values = [9.8, 10.9]),
]
forecast_dates = Date(2024,1,1):Day(1):Date(2024,1,10)
forecasts = forecast_with_nowcasts(base_model, nowcast_scenarios, forecast_dates, 100)
```
"""
function forecast_with_nowcasts(
        base_model::AutoGP.GPModel, nowcasts::AbstractVector{<:TData},
        forecast_dates, forecast_draws_per_nowcast::Int;
        inv_transformation = y -> y, n_mcmc = 0, n_hmc = 0, ess_threshold = 0.0,
        forecast_n_hmc::Union{Int, Nothing} = nothing, verbose::Bool = false
    )
    @assert !isempty(nowcasts) "nowcasts vector must not be empty"
    @assert !(n_mcmc > 0 && n_hmc == 0) "If n_mcmc > 0, n_hmc must also be > 0 for MCMC refinement"
    @assert 0.0 <= ess_threshold <= 1.0 "ess_threshold must be between 0 and 1"
    @assert forecast_n_hmc === nothing||forecast_n_hmc > 0 "forecast_n_hmc must be > 0 if specified"

    base_model_dict = Dict(base_model)
    progress = Progress(length(nowcasts); desc = "Nowcast scenarios: ", enabled = verbose)
    # Process each nowcast scenario in parallel (requires Julia 1.11+ for safe @threads nesting)
    tasks = map(nowcasts) do nowcast
        Threads.@spawn begin
            model_for_batch = AutoGP.GPModel(deepcopy(base_model_dict))
            # Add the nowcast data to the model
            AutoGP.add_data!(model_for_batch, nowcast.ds, nowcast.y)

            # Resample particles if effective sample size is below threshold
            AutoGP.maybe_resample!(
                model_for_batch, ess_threshold *
                    AutoGP.num_particles(model_for_batch)
            )

            # Optional: Refine the GP models with MCMC steps to incorporate the new data
            # into the kernel structure
            if n_mcmc > 0 && n_hmc > 0
                AutoGP.mcmc_structure!(model_for_batch, n_mcmc, n_hmc)
            elseif n_mcmc == 0 && n_hmc > 0
                AutoGP.mcmc_parameters!(model_for_batch, n_hmc)
            end

            # Generate forecasts for this nowcast scenario (verbose=false to avoid per-scenario noise)
            scenario_forecasts = forecast(
                model_for_batch, forecast_dates, forecast_draws_per_nowcast;
                inv_transformation = inv_transformation, forecast_n_hmc = forecast_n_hmc
            )

            scenario_forecasts
        end
    end

    results = map(tasks) do t
        r = fetch(t)
        next!(progress)
        r
    end
    return hcat(results...)
end
