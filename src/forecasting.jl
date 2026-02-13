"""
    forecast(model, forecast_dates, forecast_draws::Int)

Generate forecasts using the fitted `AutoGP` model.

# Arguments
- `model`: The fitted GP model.
- `forecast_dates`: A vector of dates for which to generate forecasts.
- `forecast_draws`: The number of forecast samples to draw (default: 2000).
# Returns
- A matrix of forecast samples, where each column corresponds to a sample for the respective date.
"""
function forecast(model, forecast_dates, forecast_draws::Int; inv_transformation = y -> y)
    # Convert forecast_dates to vector if it's a range
    dates_vector = collect(forecast_dates)
    dist = AutoGP.predict_mvn(model, dates_vector)
    _forecasts = rand(dist, forecast_draws)
    # Apply inverse transformation to the forecasts
    forecasts = inv_transformation.(_forecasts)
    return forecasts
end

"""
    forecast_with_nowcasts(base_model, nowcasts, forecast_dates, forecast_draws_per_nowcast;
                          inv_transformation = y -> y, n_mcmc = 0, n_hmc = 0, ess_threshold = 0.0)

Generate forecasts incorporating uncertainty from nowcast data by updating a base GP model with each nowcast scenario.

# Arguments
- `base_model`: A fitted GP model representing the baseline model (trained on historical data without nowcast period).
- `nowcasts`: A vector of nowcast data objects, each a NamedTuple with fields `ds` (dates), `y` (transformed values), and `values` (original values).
- `forecast_dates`: A vector or range of dates for which to generate forecasts.
- `forecast_draws_per_nowcast`: The number of forecast samples to draw per nowcast scenario.
- `inv_transformation`: A function to apply inverse transformation to forecasts (default: identity).
- `n_mcmc`: Number of MCMC structure steps for model refinement after adding nowcast data (default: 0). If > 0, `n_hmc` must also be > 0.
- `n_hmc`: Number of HMC parameter steps per MCMC step (default: 0). Can be > 0 even if `n_mcmc` = 0 for parameter-only updates.
- `ess_threshold`: Effective Sample Size threshold for particle resampling, as a fraction of total particles (default: 0.0, meaning don't resample).

# Returns
A matrix with dimensions `(length(forecast_dates), length(nowcasts) * forecast_draws_per_nowcast)`. Each set of `forecast_draws_per_nowcast` columns represents forecasts for one nowcast scenario, concatenated horizontally across all scenarios.

# Notes
- The `base_model` should be fitted on historical data that does not temporally overlap with the nowcast period
- Each nowcast represents a different realization of the uncertain nowcast values
- Setting `n_mcmc = 0, n_hmc > 0` performs only parameter updates without structure changes
- Setting both `n_mcmc > 0, n_hmc > 0` performs full MCMC including structure updates
- The function asserts that nowcasts is non-empty and validates MCMC parameter combinations

# Example
```julia
# Assume base_model is fitted on historical data
nowcast_scenarios = [
    (ds = [Date(2024,1,1), Date(2024,1,2)], y = [10.5, 11.2], values = [10.5, 11.2]),
    (ds = [Date(2024,1,1), Date(2024,1,2)], y = [9.8, 10.9], values = [9.8, 10.9]),
    # ... more nowcast scenarios
]
forecast_dates = Date(2024,1,1):Day(1):Date(2024,1,10) # Can overlap with nowcast dates for predictive sampling as well as forecasting
forecasts = forecast_with_nowcasts(base_model, nowcast_scenarios, forecast_dates, 100)
```
"""
function forecast_with_nowcasts(
        base_model::AutoGP.GPModel, nowcasts::AbstractVector{<:TData},
        forecast_dates, forecast_draws_per_nowcast::Int;
        inv_transformation = y -> y, n_mcmc = 0, n_hmc = 0, ess_threshold = 0.0
)
    @assert !isempty(nowcasts) "nowcasts vector must not be empty"
    @assert !(n_mcmc > 0 && n_hmc == 0) "If n_mcmc > 0, n_hmc must also be > 0 for MCMC refinement"

    # Process each nowcast scenario
    forecasts_over_nowcasts = mapreduce(hcat, nowcasts) do nowcast
        # Add the nowcast data to the model
        # NB: this work-around is necessary because of a problem with serialization of models and their deep copying
        # https://github.com/probsys/AutoGP.jl/issues/28#issuecomment-3300543503
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
            inv_transformation = inv_transformation
        )

        # Clean up the added nowcast data to restore the model data to its original state
        AutoGP.remove_data!(base_model, nowcast.ds)

        return scenario_forecasts
    end

    return forecasts_over_nowcasts
end
