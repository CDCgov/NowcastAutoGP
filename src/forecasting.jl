"""
    forecast(model, forecast_dates, forecast_draws::Int)

Generate forecast samples from a fitted `AutoGP` model.

# Arguments
- `model`: Fitted `AutoGP.GPModel`.
- `forecast_dates`: Vector or range of dates to predict.
- `forecast_draws`: Number of samples to draw.

# Keyword arguments
- `inv_transformation`: Function applied elementwise to map forecasts back to the original scale (default: identity).
- `forecast_n_hmc`: If `nothing`, draw from the current model state. If an `Int`, run that many HMC parameter steps before each draw (default: 1).

# Returns
- A matrix of samples with size `(length(forecast_dates), forecast_draws)`.
"""
function forecast(
        model::AutoGP.GPModel, forecast_dates, forecast_draws::Int;
        inv_transformation = y -> y, forecast_n_hmc::Union{Int, Nothing} = 1
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
    _forecasts = mapreduce(hcat, 1:forecast_draws) do _
        # Refine the GP models with HMC steps to incorporate the new data into
        # hyperparameters but not structure
        AutoGP.mcmc_parameters!(model, forecast_n_hmc)
        dist = AutoGP.predict_mvn(model, dates_vector)
        rand(dist)
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
- `forecast_n_hmc`: Number of HMC steps to run before each forecast draw (default: 1). If `nothing`, no HMC steps are taken before forecasting.

# Returns
- A matrix with size `(length(forecast_dates), length(nowcasts) * forecast_draws_per_nowcast)`.

# Notes
- Each scenario is added, optionally refined, forecasted, and then removed to restore the model state.
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
        inv_transformation = y -> y, n_mcmc = 0, n_hmc = 0, ess_threshold = 0.0, forecast_n_hmc::Int = 1
    )
    @assert !isempty(nowcasts) "nowcasts vector must not be empty"
    @assert !(n_mcmc > 0 && n_hmc == 0) "If n_mcmc > 0, n_hmc must also be > 0 for MCMC refinement"

    # Process each nowcast scenario
    forecasts_over_nowcasts = mapreduce(hcat, nowcasts) do nowcast
        # Add the nowcast data to the model
        AutoGP.add_data!(base_model, nowcast.ds, nowcast.y)

        # Resample particles if effective sample size is below threshold

        AutoGP.maybe_resample!(base_model, ess_threshold * AutoGP.num_particles(base_model))

        # Optional: Refine the GP models with MCMC steps to incorporate the new data
        # into the kernel structure
        if n_mcmc > 0 && n_hmc > 0
            AutoGP.mcmc_structure!(base_model, n_mcmc, n_hmc)
        elseif n_mcmc == 0 && n_hmc > 0 # Optional: Refine the GP models with HMC steps to incorporate the new data into hyperparameters but not structure
            AutoGP.mcmc_parameters!(base_model, n_hmc)
        end

        # Generate forecasts for this nowcast scenario
        scenario_forecasts = forecast(
            base_model, forecast_dates, forecast_draws_per_nowcast;
            inv_transformation = inv_transformation, forecast_n_hmc = forecast_n_hmc
        )

        # Clean up the added nowcast data to restore the model data to its original state
        AutoGP.remove_data!(base_model, nowcast.ds)

        return scenario_forecasts
    end

    return forecasts_over_nowcasts
end
