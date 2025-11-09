# Stock-Flow Consistent (SFC) Macroeconomic Models

A Python framework for building and simulating Stock-Flow Consistent macroeconomic models, based on Godley & Lavoie (2007).

## Features

- **Config-based model definition**: Define models in TOML files without writing Python code
- **Automatic equation solving**: Uses scipy's fsolve for robust numerical solutions
- **Time-series simulation**: Track variables over multiple periods
- **Automatic plotting**: Built-in visualization with subplot grids
- **Lag support**: Simple `var(-1)` notation for lagged variables
- **Type-safe state management**: Namedtuples for readable variable access

## Quick Start

### Installation

```bash
# Install dependencies
uv sync

# Install package in editable mode
uv pip install -e .
```

### Running Models

```python
from me_sfc.model import Model

# Load model from config file
model = Model(config_path="models/sim.toml")

# Simulate 100 periods
results = model.simulate(periods=100)

# Display results
print(results.tail())

# Plot all variables
model.plot()
```

Or run the example script:

```bash
uv run python examples/run_config_models.py
```

## Available Models

### SIM - Simplest Model with Government Money
- 11 equations, 11 unknowns
- Demonstrates government money creation and household wealth accumulation
- Config: `models/sim.toml`

### SIMEX - SIM with Expectations
- 13 equations, 13 unknowns
- Adds adaptive expectations to the SIM model
- Config: `models/simex.toml`

### PC - Portfolio Choice
- 10 equations, 10 unknowns
- Agents choose between holding money and interest-bearing bills
- Config: `models/pc.toml`

## Creating New Models

Create a TOML file in `models/` directory:

```toml
[metadata]
name = "Your Model"
description = "Model description"

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

Then load and run:

```python
model = Model(config_path="models/your_model.toml")
results = model.simulate(periods=100)
```

## Documentation

- **CLAUDE.md**: Detailed development guide and architecture documentation
- **examples/**: Example scripts demonstrating usage

## Testing

```bash
# Run all tests
uv run pytest tests/

# Verify models produce correct results
uv run python examples/run_config_models.py
```

## References

Godley, W., & Lavoie, M. (2007). *Monetary Economics: An Integrated Approach to Credit, Money, Income, Production and Wealth*. Palgrave Macmillan.

## License

See LICENSE file for details.
