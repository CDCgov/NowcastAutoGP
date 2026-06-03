using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()


using Markdown
md"""
# Setting GP priors with `Accessors.jl`

*Customising the AutoGP prior for epidemiological seasonality*

**CDC Center for Forecasting and Outbreak Analytics (CFA/CDC)**

`make_and_fit_model` accepts a `config` keyword — an `AutoGP.GP.GPConfig` that describes the
Gaussian-process **prior**: the distribution over kernel *structures*
(`node_dist_leaf`/`node_dist_nocp`/`node_dist_cp`) and the priors over kernel *hyperparameters*
(`prior[:gamma]`, `prior[:period]`, `prior[:wildcard]`). The default `GPConfig()` reproduces
AutoGP's out-of-the-box behaviour, but its **period prior** is centred well below an annual cycle, so
a periodic component is unlikely to land on the ~1-year seasonality respiratory diseases actually have.

`GPConfig` is an immutable struct with a nested `prior` `Dict`, which makes "change just this one
field" awkward to write by hand. [`Accessors.jl`](https://github.com/JuliaObjects/Accessors.jl)'s
`@set` macro does exactly that: it returns a *copy* with a single leaf changed and every sibling
preserved. This vignette inspects the default prior, re-centres the period prior with `@set`, and —
following the scoring approach of the [Getting started](getting-started.md) vignette — shows the
re-centred prior gives better forecasts across a few **report dates** spanning one seasonal cycle,
measured by the Continuous Ranked Probability Score (CRPS).

> `Accessors.jl` is a convenience used here in the docs; it is **not** a dependency of
> `NowcastAutoGP` itself. The same edits can be made by constructing a `GPConfig` directly.

## Loading dependencies
"""

using NowcastAutoGP
using Accessors
using CairoMakie
using Dates, Distributions, Random

Random.seed!(1234)
CairoMakie.activate!(type = "png")
nothing #hide

md"""
## Inspecting the default prior

`GPConfig()` exposes the prior as plain fields. The leaf-kernel distribution is a probability vector
over the primitive kernels, indexed `Constant=1`, `Linear=2`, `SquaredExponential=3`,
`GammaExponential=4`, `Periodic=5`:
"""

default_config = GPConfig()
default_config.node_dist_leaf

md"""
The hyperparameter priors live in a nested `Dict`; the period prior is a `LogNormal(μ, σ)` over the
periodic component's period:
"""

default_config.prior[:period]

md"""
AutoGP rescales the input time axis to `[0, 1]` internally, so this period is in *normalised* units —
a fraction of the training window. The default median period is therefore only about a fifth of the
window:
"""

exp(default_config.prior[:period][:mu]) # ≈ 0.22 of the window

md"""
## A seasonal series and a few report dates

We simulate three years of weekly observations on a **log scale** — a clear annual cycle and a gentle
upward trend, with multiplicative (log-normal) observation noise. We then imagine sitting at three
**report dates** — at roughly one, one-and-a-half and two years of history — and at each one forecast a
full year ahead. This mirrors the [Getting started](getting-started.md) workflow of scoring forecasts
across report dates as data accrue.
"""

start_date = Date(2022, 1, 1)
all_dates = collect(start_date:Week(1):(start_date + Week(52 * 3)))   # 156 weekly points ≈ 3 years
n_all = length(all_dates)
tt = 0:(n_all - 1)
log_truth = log(50.0) .+ 1.0 .* sin.(2π .* tt ./ 52) .+ 0.02 .* tt         # annual cycle (52 weeks) + trend
truth = exp.(log_truth)
observations = exp.(log_truth .+ 0.15 .* randn(n_all))


report_weeks = 51 .+ [0, 26, 52]
horizon = 52                                                        # forecast one year ahead
report_colours = [:steelblue, :darkorange, :seagreen]

fig_data = let
    fig = Figure(size = (820, 400))
    ax = Axis(
        fig[1, 1];
        xlabel = "week",
        ylabel = "value",
        title = "Synthetic weekly series",
    )
    lines!(
        ax, Dates.value.(all_dates), truth;
        color = (:black, 0.5), label = "expected",
        linestyle = :dash, linewidth = 2
    )
    scatter!(
        ax, Dates.value.(all_dates), observations;
        color = (:black, 0.8), markersize = 5, label = "observed"
    )
    vlines!(
        ax, Dates.value.(all_dates[report_weeks]);
        color = report_colours, linestyle = :dash, linewidth = 2
    )
    axislegend(ax; position = :lt)
    fig
end

md"""
## Re-centring the period prior with `@set`

AutoGP works in *normalised* time: it rescales the training window to `[0, 1]`, so a `Periodic`
kernel's period is expressed as a **fraction of the window**, and `prior[:period]` is a
`LogNormal(μ, σ)` over that fraction (its median is `exp(μ)`). Re-centring the prior is then just a
matter of `@set`-ting a new `μ` — here, a median period of a third of the window, held tightly with a
small `σ`:
"""

seasonal_example = @set GPConfig().prior[:period][:mu] = -log(3.0)
seasonal_example = @set seasonal_example.prior[:period][:sigma] = 0.1
seasonal_example.prior[:period]

md"""
`@set` returns a fresh `GPConfig`; every sibling prior is carried over unchanged — it only touched
`prior[:period]`:
"""

seasonal_example.prior[:gamma] == GPConfig().prior[:gamma]

md"""
## Forecasting from each report date

At each report date we fit two models on the data so far — one with the default prior, one with a
seasonal prior built for that window with `@set` — and forecast a year ahead. Everything except
`config` is identical; the `n_mcmc`/`n_hmc` controls pass straight through to `AutoGP.fit_smc!`. Each
fitted model is a particle ensemble, so each forecast is a full predictive *distribution* — exactly
what CRPS scores below.
"""

n_particles = 32
fit_params = (
    smc_data_proportion = 0.005,
    n_mcmc = 200,
    n_hmc = 25,
    adaptive_rejuvenation = true,
)
n_draws = 1000

Random.seed!(2026)
results = map(report_weeks) do w
    horizon_dates = all_dates[(w + 1):(w + horizon)]
    horizon_truth = observations[(w + 1):(w + horizon)]

    transformation, inv_transformation = get_transformations("positive", observations[1:w])
    train_data = create_transformed_data(
        all_dates[1:w], observations[1:w]; transformation
    )

    ## a seasonal prior for *this* window: an annual cycle is 365 days and the window spans
    ## `window_length` days, so in normalised units the period is 365/window_length → μ = log(365/window_length)
    window_length = Dates.value(all_dates[w] - all_dates[1])
    seasonal_config = @set GPConfig().prior[:period][:mu] = log(365 / window_length)
    seasonal_config = @set seasonal_config.prior[:period][:sigma] = 0.3
    seasonal_config_lin_period_prior = @set seasonal_config.node_dist_leaf = [0.0, 0.5, 0.0, 0.0, 0.5]
    default_config_lin_period_prior = @set GPConfig().node_dist_leaf = [0.0, 0.5, 0.0, 0.0, 0.5]

    default_model = make_and_fit_model(
        train_data;
        n_particles, config = GPConfig(), fit_params...
    )
    seasonal_model = make_and_fit_model(
        train_data;
        n_particles, config = seasonal_config, fit_params...
    )
    seasonal_config_lin_period_model = make_and_fit_model(
        train_data;
        n_particles, config = seasonal_config_lin_period_prior, fit_params...
    )
    default_config_lin_period_model = make_and_fit_model(
        train_data;
        n_particles, config = default_config_lin_period_prior, fit_params...
    )

    return (;
        report_week = w, horizon_dates, horizon_truth,
        default = forecast(
            default_model, horizon_dates, n_draws;
            inv_transformation
        ),
        seasonal = forecast(
            seasonal_model, horizon_dates, n_draws;
            inv_transformation
        ),
        seasonal_config_lin_period = forecast(
            seasonal_config_lin_period_model, horizon_dates, n_draws;
            inv_transformation
        ),
        default_config_lin_period = forecast(
            default_config_lin_period_model, horizon_dates, n_draws;
            inv_transformation
        ),
        default_model = default_model,
        seasonal_model = seasonal_model,
        seasonal_config_lin_period_model = seasonal_config_lin_period_model,
        default_config_lin_period_model = default_config_lin_period_model,
    )
end
nothing #hide

md"""
We forecast a year ahead from each report date under all four priors (one row each). The strong
seasonal prior (rows 3–4) is what carries the annual cycle forward and stays close to the data;
restricting the leaf kernels to Linear + Periodic (rows 2 and 4) makes little difference on its own:
"""

fig_forecasts = let
    fig = Figure(size = (920, 720))
    panels = (
        (key = :default, row = 1, title = "Default prior"),
        (key = :default_config_lin_period, row = 2, title = "Default prior with linear/periodic GP structure"),
        (key = :seasonal, row = 3, title = "Seasonal prior"),
        (key = :seasonal_config_lin_period, row = 4, title = "Seasonal prior with linear/periodic GP structure"),
    )
    for panel in panels
        ax = Axis(fig[panel.row, 1]; xlabel = "week", ylabel = "value", title = panel.title)
        scatter!(
            ax, Dates.value.(all_dates), observations;
            color = :black, label = "observations"
        )
        for (res, colour) in zip(results, report_colours)
            fc = getproperty(res, panel.key)
            fx = Dates.value.(res.horizon_dates)
            lower = [quantile(row, 0.25) for row in eachrow(fc)]
            med = [quantile(row, 0.5) for row in eachrow(fc)]
            upper = [quantile(row, 0.75) for row in eachrow(fc)]
            band!(ax, fx, lower, upper; color = (colour, 0.2))
            lines!(ax, fx, med; color = colour, linewidth = 2.5, label = "from week $(res.report_week)")
        end
        panel.row == 1 && axislegend(ax; position = :lt, nbanks = 2)
    end
    fig
end

md"""
## Scoring with CRPS

This is a probabilistic model, so we score each predictive *distribution* against the value it should
have predicted using the **Continuous Ranked Probability Score (CRPS)** — the proper scoring rule from
the [Getting started](getting-started.md) vignette (lower is better), reusing the same hand-rolled
estimator:

```math
\text{CRPS}(X, y) = \mathbb{E}[|X - y|] - \frac{1}{2}\mathbb{E}[|X_1 - X_2|]
```

"""

## Hand-rolled CRPS estimator (reproduced from the Getting started vignette).
function crps(y::Real, X::Vector{<:Real})
    n = length(X)

    ## First term: E|X - y|
    term1 = mean(abs.(X .- y))

    ## Second term: E|X_1 - X_2| over all ordered pairs
    ordered_pairwise_diffs = [abs(X[i] - X[j]) for i in 1:n for j in (i + 1):n]
    term2 = mean(ordered_pairwise_diffs)

    ## CRPS = E|X - y| - 0.5 * E|X_1 - X_2|
    return term1 - 0.5 * term2
end

## mean CRPS over a forecast horizon for a (dates × draws) forecast matrix
mean_crps(truth, fc) = mean(crps(y, collect(X)) for (y, X) in zip(truth, eachrow(fc)))

crps_by_date = map(results) do res
    (;
        report_week = res.report_week,
        default = mean_crps(res.horizon_truth, res.default),
        default_lin_period = mean_crps(res.horizon_truth, res.default_config_lin_period),
        seasonal = mean_crps(res.horizon_truth, res.seasonal),
        seasonal_lin_period = mean_crps(res.horizon_truth, res.seasonal_config_lin_period),
    )
end

md"""
Scoring confirms the visual read. The strong seasonal prior gives a markedly lower CRPS, while
restricting the leaf kernels to Linear + Periodic barely moves the score — on its own it is not
particularly beneficial. It is the period *hyperparameter* prior, not the kernel menu, doing the work:
"""

fig_scores = let
    ## one dodged bar per approach, grouped by report date
    approaches = [
        (key = :default, label = "default", colour = :tomato),
        (key = :default_lin_period, label = "default, lin+periodic leaves", colour = :goldenrod),
        (key = :seasonal, label = "seasonal", colour = :steelblue),
        (key = :seasonal_lin_period, label = "seasonal, lin+periodic leaves", colour = :seagreen),
    ]
    n = length(crps_by_date)

    xs = Int[]
    heights = Float64[]
    dodge = Int[]
    colours = Symbol[]
    for (j, approach) in enumerate(approaches), (i, row) in enumerate(crps_by_date)
        push!(xs, i)
        push!(heights, getproperty(row, approach.key))
        push!(dodge, j)
        push!(colours, approach.colour)
    end

    fig = Figure(size = (820, 430))
    ax = Axis(
        fig[1, 1];
        xticks = (1:n, ["week $(row.report_week)" for row in crps_by_date]),
        ylabel = "mean CRPS (lower is better)",
        title = "Forecast skill by report date and prior"
    )
    barplot!(ax, xs, heights; dodge = dodge, color = colours)
    Legend(
        fig[1, 2],
        [PolyElement(color = a.colour) for a in approaches],
        [a.label for a in approaches]
    )
    fig
end

md"""
Averaged over the report dates the strong seasonal prior is the clear winner; the leaf-kernel
restriction barely changes the score, with or without it:
"""

overall_crps = (;
    default = mean(row.default for row in crps_by_date),
    default_lin_period = mean(row.default_lin_period for row in crps_by_date),
    seasonal = mean(row.seasonal for row in crps_by_date),
    seasonal_lin_period = mean(row.seasonal_lin_period for row in crps_by_date),
)

md"""
## Enabling `SquaredExponential` structure

The leaf-kernel distribution is editable the same way. The default gives `SquaredExponential`
(index 3) **zero** prior mass; `@set` can spread mass evenly over `Linear`, `SquaredExponential`,
`GammaExponential` and `Periodic` (the vector must sum to 1):
"""

se_config = @set GPConfig().node_dist_leaf = [0.0, 0.25, 0.25, 0.25, 0.25]
se_config.node_dist_leaf

md"""
`se_config` can be passed to `make_and_fit_model(...; config = se_config)` exactly like the seasonal
one, and edits compose — chain further `@set` calls (or use `@reset`) to adjust the period prior *and*
the kernel distribution together.

## Summary

- `make_and_fit_model(...; config = ...)` forwards any `AutoGP.GP.GPConfig` to the model, so the full
  AutoGP prior is available without re-declaring it in `NowcastAutoGP`.
- `Accessors.@set` is a clean way to change one prior entry while preserving the rest, including deep
  edits into the nested `prior` `Dict`.
- Re-centring the period *hyperparameter* prior (`prior[:period]`) on the seasonality you expect can
  substantially improve forecasts — here the strong seasonal prior gives the lowest mean CRPS across
  the report dates.
- Editing the *structural* prior (`node_dist_leaf`) to allow only Linear + Periodic leaves made little
  difference on its own: the period prior, not the kernel menu, drove the gains. As always, score
  competing priors with CRPS (averaged over the forecasts) rather than assuming an edit will help.
"""

md"""
## Appendix: the discovered kernel structures

Each fitted model is a *weighted ensemble* of SMC particles, and every particle is a complete GP with
its own covariance-kernel structure. AutoGP exposes them directly: `particle_weights` gives each
particle's posterior weight and `covariance_kernels` its discovered kernel. Reading these out shows
which structures the search actually settled on under a given prior — a concrete look at what the prior
did. Here is the ensemble from the default-prior fit at the final report date, ordered by weight:
"""

import NowcastAutoGP.AutoGP as AGP

function show_discovered_kernels(model)
    weights = AGP.particle_weights(model)
    kernels = AGP.covariance_kernels(model)
    ## list particles from highest to lowest posterior weight
    for i in sortperm(weights; rev = true)
        println("posterior weight = ", round(weights[i]; digits = 3))
        display(kernels[i])
    end
    return nothing
end

show_discovered_kernels(results[end].default_model)

md"""
Running the same readout on the strong-seasonal-prior fit lets us compare how re-centring the period
prior reshaped the structures the search selected:
"""

show_discovered_kernels(results[end].seasonal_model)
