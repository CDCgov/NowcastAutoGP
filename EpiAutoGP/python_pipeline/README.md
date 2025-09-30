# EpiAutoGP Python Pipeline

Python preprocessing components for the EpiAutoGP epidemiological modeling pipeline.

## Overview

This package provides data preprocessing functionality for converting epidemiological surveillance data into the format required by the EpiAutoGP Julia modeling framework.

## Key Components

- **`pipeline/`** - Core Python package with data processing functions
  - `generate_test_data.py` - Generate synthetic test data for model development
  - `prep_data.py` - Data preprocessing and formatting functions  
  - `prep_ww_data.py` - Wastewater data processing
  - `utils.py` - Utility functions for PyRenew-HEW integration

## Usage

### Generate Test Data
```bash
uv run python pipeline/generate_test_data.py output_directory
```

### Development Setup
```bash
# Install dependencies
uv sync

# Run scripts
uv run python pipeline/script_name.py
```

## Dependencies

- **pyrenew-hew**: Epidemiological modeling framework
- **polars**: Fast DataFrame library for data processing
- **forecasttools**: CDC forecasting utilities
- **JAX**: Numerical computing
- **arviz**: Bayesian inference visualization

## Integration

This Python pipeline generates preprocessed data that is consumed by the Julia EpiAutoGP modeling components in the parent directory.