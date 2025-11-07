# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a stock-flow consistent (SFC) macroeconomic modeling project in Python. Models are systems of simultaneous equations solved numerically using scipy's `fsolve` nonlinear solver.

**Two Approaches Available**:
1. **Config-based models** (RECOMMENDED): Define models in simple text files (`models/*.txt`)
2. **Class-based models**: Define models as Python classes (legacy, still supported)

## Development Commands

**Package Manager**: This project uses `uv` for Python package management.

```bash
# Install dependencies
uv sync

# Install package in editable mode (required for imports to work)
uv pip install -e .

# Run config-based models (RECOMMENDED)
uv run python examples/run_config_models.py

# Run class-based models (legacy)
uv run python -m me_sfc.sim
uv run python -m me_sfc.simex
uv run python -m me_sfc.pc

# Run tests
uv run pytest tests/
```

## Config-Based Models (RECOMMENDED)

The **ConfigModel** class allows you to define models using simple text configuration files. This is the preferred approach for new models.

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

## Architecture (Class-Based Models)

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

### Config-Based (RECOMMENDED)
- **SIM** (`models/sim.txt`): Simplest model with government money - 11 equations
- **SIMEX** (`models/simex.txt`): SIM with expectations - 13 equations
- **PC** (`models/pc.txt`): Portfolio choice model - 10 equations

### Class-Based (Legacy)
- **SIM** (`src/me_sfc/sim.py`): Simplest model with government money - 11 equations
- **SIMEX** (`src/me_sfc/simex.py`): SIM with expectations - 13 equations
- **PC** (`src/me_sfc/pc.py`): Portfolio choice model - 10 equations

Both approaches produce identical results. Config-based models are easier to create and modify.
