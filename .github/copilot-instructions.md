# Copilot Instructions for NowcastAutoGP.jl

## Project Overview
NowcastAutoGP.jl is a **Julia package for forecasting with nowcasting** that combines nowcasting with Gaussian Process (GP) ensemble forecasting using AutoGP.jl. The core challenge: recent epidemiological data (last ~1-2 weeks) contains reporting delays, making "naive" forecasts systematically underestimate future values. This package works with all sorts of pure time series data which are subject to reporting delays and revisions, this can be beyond epidemiological applications.

**Key Concept**: Generate multiple nowcast scenarios for uncertain recent data, then use AutoGP's incremental inference to create forecasts for each scenario, producing uncertainty-aware predictions.

## Architecture & Data Flow

### Core Data Structure: `TData{D,F}`
- **Purpose**: Container for transformed time series with both original and transformed values
- **Fields**: `ds` (dates), `y` (transformed values), `values` (original values)
- **Constructor**: `TData(dates, values; transformation=func)` applies transformation and promotes types
- **Key Pattern**: Always maintain both original and transformed data for inverse transformations

### Main Pipeline (see `docs/vignettes/tutorial.qmd` for complete example)
1. **Transform Data**: `create_transformed_data(dates, values; transformation)` using `get_transformations()`
2. **Fit Base Model**: `make_and_fit_model(data)` on historical "confirmed" data
3. **Create Nowcasts**: Generate multiple `TData` objects representing uncertain recent data scenarios
4. **Forecast with Uncertainty**: `forecast()` temporarily adds each nowcast to model, generates forecasts, then removes it

## Transformation System (`src/transformations.jl`)

### Three Transformation Types
- **`"percentage"`**: For 0-100 bounded data → `logit((y + offset) / 100)`
- **`"positive"`**: For positive data → `log(y + offset)`
- **`"boxcox"`**: Auto-fitted Box-Cox transformation → `BoxCox_λ(y + offset)`

### Offset Logic
- Calculated via `_get_offset()`: if `min(values) == 0`, use `min(positive_values)/2`, else `0`
- **Critical**: Ensures numerical stability when data contains zeros

### Inverse Transformations
- `get_transformations()` returns `(forward, inverse)` tuple
- Box-Cox inverse uses `_inv_boxcox()` with careful edge case handling for negative λ
- Always clamp results to non-negative values: `max(result, zero(F))`

## Model Fitting Patterns (`src/make_and_fit_model.jl`)

```julia
# Standard pattern for fitting
model = make_and_fit_model(data;
    n_particles=8,           # SMC particles
    smc_data_proportion=0.1, # Data proportion per SMC step
    n_mcmc=200,             # MCMC structure steps
    n_hmc=50                # HMC parameter steps
)
```

**Key Implementation Detail**: `effective_proportion = max(smc_data_proportion, 1.0/n_train)` prevents schedule step=0 errors with small datasets.

## Forecasting with Nowcasts (`src/forecasting.jl`)

### Critical Pattern: Temporary Model Updates
```julia
# For each nowcast scenario:
AutoGP.add_data!(base_model, nowcast.ds, nowcast.y)    # Add nowcast
# Optional: AutoGP.mcmc_structure!(model, n_mcmc, n_hmc) # Refine model
forecasts = forecast(model, forecast_dates, draws)      # Generate forecasts
AutoGP.remove_data!(base_model, nowcast.ds)            # Clean up!
```

**Why This Pattern**: AutoGP models maintain internal state. Must restore original state between nowcast scenarios to avoid data accumulation.

### MCMC Parameter Combinations
- `n_mcmc=0, n_hmc=0`: No refinement (fastest)
- `n_mcmc=0, n_hmc>0`: Parameter updates only
- `n_mcmc>0, n_hmc>0`: Full structure + parameter updates
- **Invalid**: `n_mcmc>0, n_hmc=0` (asserted in code)

## Testing Philosophy

### Test Structure with TestItems.jl
- **Framework**: TestItems.jl with VS Code integration for individual test execution
- **Test Environment**: Separate `test/Project.toml` with test-specific dependencies
- **Test Runner**: `TestItemRunner.jl` with `@run_package_tests`

### Testing Patterns

#### Critical Pattern: `@testsnippet` Setup
Do not import the main module in snippets; TestItems auto-provides access. `using` and `import` won't fail but are unnecessary. Also, `using Test` and `import Test` are not needed in the `TestItems` context.
```julia
@testsnippet DataSnippet begin
    using Dates, LogExpFunctions
    # Note: Do NOT import the main module - TestItems auto-provides access

    dates = collect(Date(2024, 1, 1):Day(1):Date(2024, 1, 10))
    values = [10.0, 15.0, 12.0, 18.0, 22.0, 25.0, 20.0, 16.0, 14.0, 11.0]
    # Define all shared test data here
end

@testitem "Feature Test" setup=[DataSnippet] begin
    # Test logic only - no repetitive setup
    result = TData(dates, values; transformation = identity)
    @test result.ds == dates
end
```

#### Performance Testing Pattern
- **Fast Tests**: `(n_particles=1, n_mcmc=5, n_hmc=3)` for CI/local development
- **Reproducibility**: `Random.seed!(42)` for synthetic data generation
- **VS Code Integration**: Individual test execution via Test Explorer

### Essential Test Categories
- Transformation round-trip accuracy with edge cases (zeros, negatives)
- Model fitting with different data sizes/transformations
- Nowcast integration (temporary model state changes)
- AutoGP.jl interface compliance (add_data!/remove_data! patterns)

## Development Workflow

### Documentation System (`docs/`)
- **Framework**: Documenter.jl with Material Design theme for GitHub Pages
- **Build Process**: `julia --project=docs docs/make.jl` (local), GitHub Actions (deployment)
- **Tutorial System**: Quarto files (`docs/vignettes/tutorial.qmd`) → markdown for Documenter
- **Content Strategy**: Public health focus with epidemiological examples (NHSN COVID-19 data)

#### Key Documentation Files
- `docs/make.jl` - Documenter.jl configuration
- `docs/src/index.md` - CFA colour styled landing page
- `docs/src/api.md` - Auto-generated (by Documenter.jl) API reference
- `docs/src/assets/material-theme.css` - Custom Material Design styling

#### Documentation Development Pattern
```julia
# Local documentation build
cd docs/
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project make.jl
```

### Code Quality & CI/CD
- **Testing**: Multi-platform CI (Ubuntu, Windows, macOS) on Julia 1.10 + latest
- **Coverage**: Codecov integration on Ubuntu/x64/Julia-latest only
- **Documentation**: Auto-deployed to GitHub Pages on main branch pushes

## Key Dependencies & Interfaces

### AutoGP.jl Integration Points
- `AutoGP.GPModel(ds, y; n_particles)` - model creation
- `AutoGP.fit_smc!(model; schedule, n_mcmc, n_hmc)` - fitting with SMC
- `AutoGP.add_data!/remove_data!` - temporary data updates
- `AutoGP.predict_mvn(model, dates)` - prediction distribution
- **Important**: Models are stateful - always clean up temporary additions

### Domain-Specific Considerations
- **CDC Context**: Government repository with extensive compliance documentation
- **Epidemiological Focus**: Designed for surveillance data with reporting delays
- **Public Health Applications**: Hospital admissions, ED visits, case counts with known revision patterns
- **Content Standards**: Professional government tone, epidemiological examples, accessibility compliance

## Development Principles

### TestItems.jl Best Practices
- **Module Access**: TestItems auto-provides module access - never import the main module in test snippets
- **Data Setup**: Use `@testsnippet` for all shared test data and external imports
- **Test Focus**: Each `@testitem` should test one specific aspect with minimal setup
- **VS Code Integration**: Tests appear individually in Test Explorer for independent execution

### Documentation Standards
- **Theme**: Material Design with CDC color scheme (`--md-primary-color: #1976d2`)
- **Content Focus**: Lead with public health mission, include realistic surveillance examples
- **Tutorial Pattern**: Quarto files with epidemiological context → Documenter.jl integration
- **Build Artifacts**: `docs/build/` gitignored, source assets tracked
