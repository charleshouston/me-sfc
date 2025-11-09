# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a stock-flow consistent (SFC) macroeconomic modeling project in Python. Models are systems of simultaneous equations solved numerically using scipy's `fsolve` nonlinear solver.

**Models are defined in TOML configuration files** (`models/*.toml`) that specify equations, parameters, exogenous variables, and initial values.

## Development Commands

**Package Manager**: This project uses `uv` for Python package management.

```bash
# Install dependencies
uv sync

# Install package in editable mode (required for imports to work)
uv pip install -e .

# Run all models
uv run python examples/run_config_models.py

# Run tests
uv run pytest tests/
```

## Config-Based Models

The **Model** class allows you to define models using TOML configuration files.

### Quick Start

```python
from me_sfc.model import Model

# Load and run a model
model = Model(config_path="models/sim.toml")
results = model.simulate(periods=100)
model.plot()
```

### Config File Format

Config files use TOML format with standard sections:

```toml
[metadata]
name = "SIM"
description = "Simplest model with government money"

[equations]
y = "c + g"
yd = "y - t"
c = "c0 * yd + c1 * h(-1)"
h = "h(-1) + yd - c"
t = "theta * y"

[parameters]
c0 = 0.6
c1 = 0.4
theta = 0.2

[exogenous]
g = 20.0

[initial]
y = 0.0
h = 0.0
```

**Key Features**:
- **Standard TOML**: Uses Python stdlib tomllib parser
- **Lag notation**: Use `var(-1)` to reference previous period values
- **Sections**: equations, parameters, exogenous, initial, metadata (optional)
- **All Model features**: Inherits `simulate()`, `plot()`, `get_results()` from base class

### Available Config Models

- `models/sim.toml`: Simplest model with government money (11 equations)
- `models/simex.toml`: SIM with expectations (13 equations)
- `models/pc.toml`: Portfolio choice model (10 equations)

### Creating New Config Models

1. Create a new `.toml` file in `models/` directory
2. Define sections: equations, parameters, exogenous, initial
3. Load with `Model(config_path="models/your_model.toml")`

**Example**: See `examples/run_config_models.py` for complete usage examples.

## Implementation Details

### Internal Architecture

The Model class uses a hybrid pattern internally:
- **Namedtuples** for type-safe state management (readable variable access)
- **Lists** for time-series storage and lagged variable access
- **TOML parsing** using Python's stdlib tomllib parser

Variables are automatically detected from equation keys in the TOML file and converted to a State namedtuple. The `var(-1)` lag notation is converted to dictionary lookups before evaluation.

### Core Methods

The Model class provides:
- `update()`: Solves equations using scipy's fsolve
- `get_results()`: Converts state history to pandas DataFrame
- `simulate(periods)`: Runs multiple periods
- `plot()`: Automatic visualization with subplot grid layout

## Available Models

- **SIM** (`models/sim.toml`): Simplest model with government money - 11 equations
- **SIMEX** (`models/simex.toml`): SIM with expectations - 13 equations
- **PC** (`models/pc.toml`): Portfolio choice model - 10 equations
