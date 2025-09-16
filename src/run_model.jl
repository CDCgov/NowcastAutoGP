using EpiAutoGP, EpiMech
using CSV, DataFramesMeta, Dates, Plots
using LogExpFunctions: logit, logistic
using BoxCox: BoxCoxTransformation, confint, fit

# Get ARGS
extra_args = (
    ["--smc_data_proportion"], Dict(
        :help => "Proportion of data to use for Sequential Monte Carlo sampling",
        :arg_type => Float64,
        :default => 0.05
    ),
    ["--transform_name"], Dict(
        :help => "Transformation applied to data",
        :arg_type => String,
        :default => "boxcox"
    ),
    ["--n_mcmc"], Dict(
        :help => "Number of MCMC samples",
        :arg_type => Int,
        :default => 200
    ),
    ["--n_hmc"], Dict(
        :help => "Number of HMC samples",
        :arg_type => Int,
        :default => 50
    ),
    ["--n_forecast_draws"], Dict(
        :help => "Number of forecast draws",
        :arg_type => Int,
        :default => 2000
    ),
    ["--n_particles"], Dict(
        :help => "Number of particles for SMC",
        :arg_type => Int,
        :default => 8
    ),
    ["--n_redact"], Dict(
        :help => "Number of data points to redact for nowcasting",
        :arg_type => Int,
        :default => 1
    ),
    ["--n_ahead"], Dict(
        :help => "Number of weeks to forecast ahead",
        :arg_type => Int,
        :default => 8
    ),
    ["--yscale"], Dict(
        :help => "Output plot yscale, either :log10 or :identity",
        :arg_type => Symbol,
        :default => :log10
    )
)

settings = basic_mech_experiment_positional_arg_settings()
add_args!(settings; extra_args)
parsed_args = parse_args(ARGS, settings)

experiment_path = parsed_args["experiment_path"]
@info "Running model in directory: $experiment_path"
model_id = parsed_args["model_id"]
@info "Model ID: $model_id"
plotname = parsed_args["plotname"]
@info "Plot name: $plotname"
forecast_df_name = parsed_args["forecast_df_name"]
@info "Forecast DataFrame name: $forecast_df_name"
vintage_data_name = parsed_args["vintage_data_name"]
@info "Vintage data name: $vintage_data_name"
full_data_name = parsed_args["full_data_name"]
@info "Full data name: $full_data_name"
loc_str = parsed_args["loc_str"]
@info "Location string: $loc_str"
smc_data_proportion = parsed_args["smc_data_proportion"]
transform_name = parsed_args["transform_name"]
n_mcmc = parsed_args["n_mcmc"]
n_hmc = parsed_args["n_hmc"]
forecast_draws = parsed_args["n_forecast_draws"]
n_redact = parsed_args["n_redact"]
n_ahead = parsed_args["n_ahead"]
n_particles = parsed_args["n_particles"]
yscale = parsed_args["yscale"]

horizon_0 = 1 - n_redact # Horizon 0 is the last data point before forecasting

# Change working directory to experiment_path
cd(experiment_path)

# Load and select data
# If in percentage mode, we only want to use positive values
@info "Loading data for $loc_str"

vintaged_df = get_targetdata_from_csv(vintage_data_name, loc_str)
full_df = get_targetdata_from_csv(full_data_name, loc_str)

reference_date = vintaged_df.date[end]
pathogen = vintaged_df.pathogen[1]
signal = vintaged_df.signal[1]

# Define function to get transformation functions
function get_transformationtions(transform_name, data_values)
    if transform_name == "percentage"
        @info "Using percentage transformation"
        return (y -> logit(y / 100), y -> logistic(y) * 100)
    elseif transform_name == "positive"
        offset = minimum(data_values[data_values .> 0]) / 2 # Half the minimum positive value for stability
        @info "Using positive transformation with offset = $offset"
        return (y -> log(y + offset), y -> max(exp(y) - offset, 0.0))
    elseif transform_name == "boxcox"
        offset = minimum(data_values[data_values .> 0]) / 2 # Half the minimum positive value for stability
        bc = fit(BoxCoxTransformation, data_values .+ offset)
        λ = bc.λ
        if λ ≈ 0
            @info "Using log transformation due to Box-Cox finding λ ≈ 0"
            return (y -> log(y + offset), y -> max(exp(y) - offset, 0.0))
        else
            @info "Using Box-Cox transformation with λ = $λ"
            # Robust inverse Box-Cox transformation that handles negative λ and edge cases
            inv_boxcox = function(y)
                lambda_y_plus_1 = λ * y + 1

                # Handle edge cases based on λ sign and lambda_y_plus_1 value
                if λ > 0
                    # Standard case: ensure lambda_y_plus_1 > 0
                    safe_value = max(lambda_y_plus_1, 1e-10)
                    result = safe_value^(1/λ) - offset
                elseif λ < 0
                    # Negative λ case: more careful handling
                    if lambda_y_plus_1 > 1e-10
                        # Normal case: lambda_y_plus_1 is sufficiently positive
                        result = lambda_y_plus_1^(1/λ) - offset
                    else
                        # Edge case: lambda_y_plus_1 ≤ 0 or very small
                        # For negative λ, (small positive)^(negative) = very large number
                        # We'll clamp this to avoid numerical explosion
                        if lambda_y_plus_1 <= 0
                            # Reduce to zero result since lambda_y_plus_1 == 0 ⟹ x == 0 when λ ≠ 0
                            # We collect this as probability mass at zero
                            result = 0.0
                        else
                            # lambda_y_plus_1 is very small but positive
                            clamped_result = lambda_y_plus_1^(1/λ)
                            # Clamp extremely large values to reasonable bounds
                            max_reasonable = 1000 * maximum(data_values)  # 1000x the max observed value
                            result = min(clamped_result, max_reasonable) - offset
                        end
                    end
                else
                    # λ ≈ 0 case should have been handled above, but just in case
                    result = exp(y) - offset
                end

                return max(result, 0.0)  # Ensure non-negative output
            end

            return (y -> bc(y + offset), inv_boxcox)
        end
    else
        error("Unknown transform_name: $transform_name")
    end
end

# Do forecasting on transformed data
transformation,
inv_transformation = get_transformationtions(transform_name, vintaged_df.value)

data = create_transformed_data(vintaged_df; transformation, n_redact) # redact last entry for "nowcasting"
n_full = size(full_df.date, 1)
n_forecast = length(data.ds) + n_ahead
full_data = create_transformed_data(full_df; transformation)

@info "Fitting model for $loc_str"
model = make_and_fit_model(
    data; n_particles, smc_data_proportion, n_mcmc, n_hmc, shuffle = true,
    adaptive_resampling = false, verbose = false, check = false)

@info "Forecasting for $loc_str"
future_dates = data.ds[end] .+ (1:n_ahead) .* Week(1)
forecast_dates = vcat(data.ds, future_dates)

forecasts = forecast(model, forecast_dates, forecast_draws; inv_transformation)

# Plot the model
@info "Plotting model for $loc_str"

function set_output_args(signal, loc_str, model_id, pathogen; yscale, full_data)
    @assert yscale in (:log10, :identity) "yscale must be :log10 or :identity"
    ylims = yscale == :log10 ?
            (0.5 * minimum(full_data.values[full_data.values .> 0]),
        1.1 * maximum(full_data.values)) :
            :auto
    if signal == "confirmed_admissions_covid_ew_prelim"
        return (ylims = ylims,
            plot_title = "$(loc_str): Confirmed $(pathogen) Admissions ($(model_id))",
            plot_ylabel = "Weekly $(pathogen) count",
            target_str = "wk inc covid hosp")
    elseif signal == "pct_ed_visits_covid"
        return (ylims = ylims,
            plot_title = "$(loc_str): $(pathogen) ED % visits ($(model_id))",
            plot_ylabel = "% ED visits",
            target_str = "wk inc covid prop ed visits")
    else
        error("Unknown signal: $signal")
    end
end

# Get output args based on signal
output_args = set_output_args(
    signal, loc_str, model_id, pathogen; yscale = yscale, full_data = full_data)

fig = plot_and_save_forecast(
    forecast_dates, forecasts; title = output_args.plot_title, ylabel = output_args.plot_ylabel,
    plot_from_date = data.ds[end] - Year(2), data, full_data,
    plotname = plotname, save = true, local_dir = "plots", yscale = yscale,
    ylims = output_args.ylims)

@info "Evaluating model for $loc_str"

eval_forecast_df = collate_and_save_forecast(
    forecasts[(end - n_ahead + 1):end, :]; target_str = output_args.target_str,
    future_dates, horizon_0, model_id, reference_date, loc_str,
    full_df, forecast_df_name, save = true, local_dir = "forecast")
