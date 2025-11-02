# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a stock-flow consistent (SFC) macroeconomic modeling project in Python. Models are systems of simultaneous equations solved numerically using scipy's `fsolve` nonlinear solver.

## Development Commands

**Package Manager**: This project uses `uv` for Python package management.

```bash
# Install dependencies
uv sync

# Run individual models directly
uv run python -m me_sfc.sim
uv run python -m me_sfc.simex

# Run tests
uv run pytest tests/
uv run python tests/test_plotting.py  # Run specific test file
```

## Architecture

### Hybrid Pattern: Namedtuples + Lists

Models use a **hybrid data structure** combining namedtuples for type safety with lists for time-series storage:

```python
# Define state structure (class-level)
State = namedtuple('State', ['Y', 'C', 'I', ...])

# Initialize with list of States
self.x = [self.State(*np.zeros(n))]

# Access in equations
current = self.State(*x)  # Current period (being solved)
prev = self.x[-1]         # Previous period (from history)

# Readable variable access (no magic indices!)
eq = current.C - (self.c0 * current.Y + self.c1 * prev.H)
```

**Why this pattern**:
- Namedtuples provide readable variable access (e.g., `current.Y` instead of `x[0]`)
- Lists enable time-series storage and access to lagged variables
- Type safety: prevents index errors and makes equations self-documenting

### Model Base Class

All models inherit from `Model` (src/me_sfc/model.py) and must implement:

1. **Class-level State namedtuple**: Defines the model's state structure
2. **`_equations(x)`**: System of equations as residuals that should equal zero

The base class provides:
- `update()`: Solves equations using fsolve and converts solution to State namedtuple
- `get_results()`: Converts State history to pandas DataFrame
- `simulate(periods)`: Runs multiple periods and returns results DataFrame
- `plot()`: Automatic visualization with subplot grid layout

### Equation Formulation Rules

1. **All equations are residuals**: Write as `expression - target = 0`
   ```python
   eq1 = current.C_s - current.C_d  # Supply equals demand
   eq2 = current.Y - (current.C + current.G)  # Income identity
   ```

2. **Access previous period via `prev`**: Use `self.x[-1]` for lagged variables
   ```python
   prev = self.x[-1]
   eq = current.H - (prev.H + current.Y - current.C)  # Wealth accumulation
   ```

3. **Unpack solution using State**: Convert numpy array to namedtuple
   ```python
   current = self.State(*x)
   ```

### Creating New Models

```python
from me_sfc.model import Model
from collections import namedtuple
import numpy as np
import pandas as pd

class YourModel(Model):
    # 1. Define state structure
    State = namedtuple('State', ['Y', 'C', 'I', ...])

    # 2. Initialize parameters and state
    def __init__(self, param1, param2):
        self.param1 = param1
        self.x = [self.State(*np.zeros(n))]

    # 3. Define system of equations
    def _equations(self, x):
        current = self.State(*x)
        prev = self.x[-1]
        # Return list of residuals
        return [eq1, eq2, ...]

    # get_results() is inherited from base class
```

## Existing Models

- **SIM** (src/me_sfc/sim.py): Simplest model with government money - 11 equations, 11 unknowns
- **SIMEX** (src/me_sfc/simex.py): SIM with expectations - 13 equations, 13 unknowns

Each model file documents its equations, exogenous parameters, and behavioral assumptions in comments.
