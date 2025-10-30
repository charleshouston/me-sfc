# Plotting Functionality Design

**Date:** 2025-10-30
**Status:** Approved

## Overview

Add plotting functionality to visualize time series results from SFC macroeconomic models. The solution must work for all current and future models in the codebase.

## Requirements

- Visualize all 11 variables from SIM model (and variable count for future models)
- Use separate subplots for each variable
- Support both interactive display and file saving
- Extensible to future models without code duplication

## Architecture

### Base Model Class Pattern

Create an abstract base class `Model` that all SFC models inherit from. This provides:

- **Abstract method `get_results()`**: Returns dictionary mapping variable names to time series arrays
- **Concrete method `plot()`**: Generates subplot visualizations from results dictionary
- **Concrete method `simulate()`**: Can be inherited or overridden by subclasses

**New file:** `model.py`

**Rationale:** This approach ensures consistency across all models, eliminates code duplication, and provides plotting functionality automatically to any new model that inherits from Model.

### Refactoring Existing SIM Model

Modify `sim.py`:
- Change `class SIM:` to `class SIM(Model):`
- No changes to existing `get_results()` or `simulate()` methods needed
- SIM already implements the required interface

## Plotting Method Interface

```python
def plot(self, save_path=None, show=True, figsize=None):
    """Plot all variables as time series in separate subplots.

    Args:
        save_path: Optional file path to save figure (e.g., "figures/results.png")
        show: Boolean to display plot interactively (default True)
        figsize: Optional tuple (width, height) for figure size, auto-calculated if None
    """
```

### Layout Strategy

- Automatic grid calculation: `cols = min(3, n_vars)`, `rows = ceil(n_vars / cols)`
- One subplot per variable showing time series
- Common x-axis (period number) across all subplots
- Clear titles showing variable names
- Axis labels on edge subplots only to reduce clutter
- Grid lines for readability

### Default Figure Sizing

- Auto-calculated: `figsize = (15, 3 * rows)` gives ~3 inches height per row
- Override via parameter for custom sizing

## Error Handling

- **Empty results:** Raise informative error if results dict is empty or contains no time periods
- **File path:** Create parent directories automatically using `os.makedirs()`
- **Grid indexing:** Handle both 1D and 2D axes arrays for single and multiple subplots
- **Format support:** Leverage matplotlib's automatic format detection from file extension

## Dependencies

Add to `pyproject.toml`:
- `matplotlib` (for visualization)

Install via: `uv add matplotlib`

## Usage Examples

### Interactive Display

```python
model = SIM(c0=0.6, c1=0.4, theta=0.2, g0=20, W=1)
results = model.simulate(periods=100)
model.plot()
```

### Save to File

```python
model.plot(save_path="figures/sim_baseline.png", show=False)
```

### Both Display and Save

```python
model.plot(save_path="figures/sim_baseline.png", show=True)
```

### Future Models

```python
class SIMEX(Model):
    def update(self):
        # Implement equation system
        pass

    def get_results(self):
        # Return dict of time series
        pass

# Plotting works automatically
model = SIMEX(...)
model.simulate(100)
model.plot()  # No additional code needed
```

## Implementation Tasks

1. Add matplotlib dependency
2. Create `model.py` with abstract base class
3. Implement `plot()` method with grid layout logic
4. Refactor `sim.py` to inherit from Model
5. Update `sim.py` main block to demonstrate plotting
6. Test with various variable counts and configurations

## Non-Goals

- Custom plot styling/themes (use matplotlib defaults)
- Interactive plot controls beyond matplotlib's built-in toolbar
- Comparison plotting between multiple model runs (future enhancement)
- Statistical analysis or summary statistics in plots
