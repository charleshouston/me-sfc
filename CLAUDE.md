# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a stock-flow consistent (SFC) macroeconomic modeling project implemented in Python. The project implements formal economic models as systems of simultaneous equations that are solved numerically using scipy's `fsolve` nonlinear solver.

## Development Commands

**Package Manager**: This project uses `uv` for Python package management.

```bash
# Install dependencies
uv sync

# Run the main script
uv run python main.py

# Run the SIM model
uv run python sim.py
```

## Architecture

### Model Structure

The codebase follows a class-based architecture where each economic model is implemented as a separate class:

- **sim.py**: Contains the `SIM` class (Simplest Model with Government Money)
  - Models a closed economy with government, households, and 11 endogenous variables
  - Uses 11 simultaneous equations to determine equilibrium values each period
  - Implements dynamic simulation via the `update()` method

### SIM Model Design Pattern

Each model class follows this pattern:

1. **Initialization (`__init__`)**:
   - Sets exogenous parameters (constants like tax rates, propensities to consume)
   - Initializes solution vector `self.x` as a list of arrays, where `self.x[-1]` contains previous period values

2. **Update Method (`update()`)**:
   - Defines a nested function `f(x)` containing the system of equations
   - Each equation is expressed as a residual that should equal zero at equilibrium
   - Accesses lagged values via `self.x[-1]` (e.g., `H_h_prev = self.x[-1][5]`)
   - Uses `fsolve` with previous period solution as initial guess
   - Appends new solution to `self.x` history

### Key Implementation Details

- **Solution Storage**: Solutions are stored as a list of numpy arrays, enabling time-series analysis and access to lagged variables
- **Equation Formulation**: All equations are written as residuals (expression - target = 0) for the nonlinear solver
- **Variable Ordering**: Variables must maintain consistent ordering across the solution vector (e.g., in SIM: Y, YD, T_d, T_s, H_s, H_h, G_s, C_s, C_d, N_s, N_d)

## Model Equations

Each model file contains detailed comments documenting:
- Number of equations and unknowns
- Exogenous (policy/parameter) variables
- Behavioral equations (consumption, investment, etc.)
- Accounting identities (budget constraints, national income identity)
- Market clearing conditions

These comments are essential for understanding and extending the models.

## Dependencies

- **numpy**: Array operations and numerical computing
- **scipy**: Nonlinear equation solving (`fsolve`)

## Plotting Functionality

All models inherit from the `Model` base class in `model.py`, which provides automatic plotting capabilities.

### Usage

```python
# Run simulation
model = SIM(c0=0.6, c1=0.4, theta=0.2, g0=20, W=1)
results = model.simulate(periods=100)

# Display plots interactively
model.plot()

# Save plots to file
model.plot(save_path="figures/results.png", show=False)

# Both display and save
model.plot(save_path="figures/results.png", show=True)
```

### Plot Features

- Automatic subplot grid layout (up to 3 columns)
- One subplot per variable showing time series evolution
- Grid lines and clear axis labels
- High-resolution output (300 DPI) for saved figures
- Supports PNG, PDF, SVG formats (detected from file extension)

### Creating New Models

New models should inherit from `Model` and implement:
- `update()`: Update state for one time step
- `get_results()`: Return dictionary mapping variable names to numpy arrays

Plotting functionality is inherited automatically.
