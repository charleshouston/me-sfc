# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a stock-flow consistent (SFC) macroeconomic modeling project in Python. Models are systems of simultaneous equations solved numerically using scipy's `fsolve` nonlinear solver.

**Models are defined in simple text configuration files** (`models/*.txt`) that specify equations, parameters, exogenous variables, and initial values.

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

The **ConfigModel** class allows you to define models using simple text configuration files.

### Quick Start

```python
from me_sfc.config_model import ConfigModel

# Load and run a model
model = ConfigModel(config_path="models/sim.txt")
results = model.simulate(periods=100)
model.plot()
```

### Config File Format

Config files use section markers (`######`) to organize model specifications:

```
###### Equations
y = c + g
yd = y - t
c = c0 * yd + c1 * h(-1)
h = h(-1) + yd - c
t = theta * y

###### Parameters
c0 = 0.6
c1 = 0.4
theta = 0.2

###### Exogenous
g = 20

###### Initial
y = 0
h = 0
```

**Key Features**:
- **Lag notation**: Use `var(-1)` to reference previous period values
- **Automatic variable detection**: Variables extracted from equation left-hand sides
- **Safe evaluation**: Restricted namespace prevents malicious code
- **All Model features**: Inherits `simulate()`, `plot()`, `get_results()` from base class

### Available Config Models

- `models/sim.txt`: Simplest model with government money (11 equations)
- `models/simex.txt`: SIM with expectations (13 equations)
- `models/pc.txt`: Portfolio choice model (10 equations)

### Creating New Config Models

1. Create a new `.txt` file in `models/` directory
2. Define sections: Equations, Parameters, Exogenous, Initial
3. Load with `ConfigModel(config_path="models/your_model.txt")`

**Example**: See `examples/run_config_models.py` for complete usage examples.

## Implementation Details

### Internal Architecture

The ConfigModel class uses a hybrid pattern internally:
- **Namedtuples** for type-safe state management (readable variable access)
- **Lists** for time-series storage and lagged variable access
- **Restricted namespace** for safe equation evaluation

Variables are automatically detected from equation left-hand sides and converted to a State namedtuple. The `var(-1)` lag notation is converted to dictionary lookups before evaluation.

### Model Base Class

ConfigModel inherits from the abstract `Model` base class (src/me_sfc/model.py), which provides:
- `update()`: Solves equations using scipy's fsolve
- `get_results()`: Converts state history to pandas DataFrame
- `simulate(periods)`: Runs multiple periods
- `plot()`: Automatic visualization with subplot grid layout

## Available Models

- **SIM** (`models/sim.txt`): Simplest model with government money - 11 equations
- **SIMEX** (`models/simex.txt`): SIM with expectations - 13 equations
- **PC** (`models/pc.txt`): Portfolio choice model - 10 equations
