# EpiAutoGP Joint Julia/Python Project Structure

This project combines Julia modeling with Python preprocessing in a safe, organized structure.

## Directory Structure

```
EpiAutoGP/
├── src/                    # Julia source code (NowcastAutoGP.jl integration)
├── test/                   # Julia tests  
├── Project.toml            # Julia project dependencies
├── run.jl                  # Julia main entry point
├── python_pipeline/        # Python preprocessing components
│   ├── pipeline/           # Python package for data preprocessing  
│   ├── pyproject.toml      # Python dependencies and build config
│   ├── uv.lock            # Python dependency lock file
│   └── .venv/             # Python virtual environment
├── end-to-end/            # Integration examples
└── test_output/           # Generated test data
```

## Usage

### Python Preprocessing
```bash
cd python_pipeline
uv run python pipeline/generate_test_data.py ../test_output
```

### Julia Modeling  
```bash
julia --project=. run.jl --json-input test_output/... --output-dir results/
```

## Safety Features

- **Isolated Python Environment**: All Python code, dependencies, and virtual environment are contained in `python_pipeline/`
- **Protected Julia Code**: Julia `src/` folder is at the root level, protected from Python build processes
- **Clear Separation**: No risk of Python packaging tools interfering with Julia source code
- **Independent Builds**: Each language component can be built/tested independently

## Development Workflow

1. **Data Generation**: Use Python scripts in `python_pipeline/` to create test data
2. **Modeling**: Use Julia code in `src/` to run EpiAutoGP models on the preprocessed data  
3. **Integration**: Use `end-to-end/` examples to test the complete pipeline