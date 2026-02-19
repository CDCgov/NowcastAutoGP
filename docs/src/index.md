# NowcastAutoGP.jl

*Centers for Disease Control and Prevention • Center for Forecasting and Outbreak Analytics*

**Automated Gaussian Process model discovery for time series data with significant on-going revisions**


## About NowcastAutoGP.jl

**NowcastAutoGP.jl** is a Julia package for combining **nowcasting** of epidemiological time series data with **forecasting** using an ensemble of Gaussian process (GP) models.
The package was developed for the [CDC Center for Forecasting and Outbreak Analytics (CFA)](https://www.cdc.gov/forecasting/index.html) to support real-time situational awareness and epidemiological forecasting.

The basic idea behind this package is to use the incremental fitting capabilities of [`AutoGP.jl`](https://github.com/probsys/AutoGP.jl) to batch forecasts over probababilistic nowcasts of recent data points. In this way, `NowcastAutoGP.jl` is able to account for the uncertainty in recent data points that are still being revised, while also leveraging the flexibility and scalability of Gaussian processes for forecasting.

The main upside of this approach is that its flexibility allows us to be agnostic to the nowcasting method used, as long as it can produce generative samples over the distribution of recent data points; noting that point estimate nowcasts can also be used as one-sample degenerate distributions. However, this does mean that:

1. The quality of the nowcasts will impact the quality of the forecasts.
2. The nowcasting and forecasting models are not jointly inferred, which may lead to suboptimal performance compared to a fully Bayesian approach in some circumstances; for example, if the nowcasting model is poorly specified.

## Installation

```julia
# Standard installation
using Pkg
Pkg.add(url="https://github.com/CDCgov/NowcastAutoGP.jl")
```

## Getting Started

- **[Getting started example](vignettes/getting-started.md)**: A getting started guide demonstrating basic usage for combining forecasting and nowcasting on NHSN covid-19 hospitalization data.

## API Reference and Resources

- **[API Documentation](api.md)**
