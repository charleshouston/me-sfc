# TOML Config Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace custom parser with stdlib TOML parser and merge model.py + config_model.py into single class.

**Architecture:** Convert model config files from custom section format to TOML, replace 115-line custom parser with ~30-line tomllib wrapper, consolidate Model + ConfigModel into single concrete class.

**Tech Stack:** Python 3.11+ stdlib tomllib, existing scipy/numpy/pandas/matplotlib stack

---

## Task 1: Convert SIM Model to TOML

**Files:**
- Create: `models/sim.toml`
- Reference: `models/sim.txt` (for conversion)

**Step 1: Create sim.toml with TOML format**

```bash
cat > models/sim.toml << 'EOF'
[metadata]
name = "SIM"
description = "Simplest Model with Government Money"
reference = "Godley & Lavoie (2007)"

[equations]
C_s = "C_d"
G_s = "g0"
T_s = "T_d"
N_s = "N_d"
YD = "W * N_s - T_s"
T_d = "theta * W * N_s"
C_d = "c0 * YD + c1 * H_h(-1)"
H_s = "H_s(-1) + g0 - T_d"
H_h = "H_h(-1) + YD - C_d"
Y = "C_s + G_s"
N_d = "Y / W"

[parameters]
c0 = 0.6
c1 = 0.4
theta = 0.2

[exogenous]
g0 = 20.0
W = 1.0

[initial]
Y = 0.0
YD = 0.0
T_d = 0.0
T_s = 0.0
H_s = 0.0
H_h = 0.0
G_s = 0.0
C_s = 0.0
C_d = 0.0
N_s = 0.0
N_d = 0.0
EOF
```

**Step 2: Verify TOML syntax**

Run: `python -c "import tomllib; tomllib.load(open('models/sim.toml', 'rb'))"`
Expected: No errors (silent success)

**Step 3: Commit**

```bash
git add models/sim.toml
git commit -m "feat: add SIM model in TOML format"
```

---

## Task 2: Convert SIMEX Model to TOML

**Files:**
- Create: `models/simex.toml`
- Reference: `models/simex.txt`

**Step 1: Create simex.toml with TOML format**

```bash
cat > models/simex.toml << 'EOF'
[metadata]
name = "SIMEX"
description = "SIM with Expectations"
reference = "Godley & Lavoie (2007)"

[equations]
C_s = "C_d"
G_s = "g0"
T_s = "T_d"
N_s = "N_d"
YD = "W * N_s - T_s"
YDe = "YD(-1)"
T_d = "theta * W * N_s"
C_d = "c0 * YDe + c1 * H_h(-1)"
H_s = "H_s(-1) + g0 - T_d"
H_d = "H_h"
H_h = "H_h(-1) + YD - C_d"
Y = "C_s + G_s"
N_d = "Y / W"

[parameters]
c0 = 0.6
c1 = 0.4
theta = 0.2

[exogenous]
g0 = 20.0
W = 1.0

[initial]
Y = 0.0
YD = 0.0
YDe = 0.0
T_d = 0.0
T_s = 0.0
H_s = 0.0
H_d = 0.0
H_h = 0.0
G_s = 0.0
C_s = 0.0
C_d = 0.0
N_s = 0.0
N_d = 0.0
EOF
```

**Step 2: Verify TOML syntax**

Run: `python -c "import tomllib; tomllib.load(open('models/simex.toml', 'rb'))"`
Expected: No errors (silent success)

**Step 3: Commit**

```bash
git add models/simex.toml
git commit -m "feat: add SIMEX model in TOML format"
```

---

## Task 3: Convert PC Model to TOML

**Files:**
- Create: `models/pc.toml`
- Reference: `models/pc.txt`

**Step 1: Create pc.toml with TOML format**

```bash
cat > models/pc.toml << 'EOF'
[metadata]
name = "PC"
description = "Portfolio Choice Model"
reference = "Godley & Lavoie (2007)"

[equations]
Y = "C + g0"
YD = "Y - T + r0 * B_h(-1)"
T = "theta * Y"
V = "V(-1) + YD - C"
C = "c0 * YD + c1 * V(-1)"
B_h = "V * (b0 + b1 * r0) - b2 * YD"
H_h = "V - B_h"
B_s = "B_h + B_cb"
H_s = "H_h"
B_cb = "B_s - B_h"

[parameters]
c0 = 0.6
c1 = 0.4
b0 = 0.4
b1 = 0.2
b2 = 0.1
theta = 0.2

[exogenous]
g0 = 20.0
r0 = 0.025

[initial]
Y = 86.48648648648648
YD = 69.18918918918919
T = 17.297297297297298
V = 86.48648648648648
C = 66.48648648648648
B_h = 64.86486486486487
H_h = 21.62162162162162
B_s = 64.86486486486487
H_s = 21.62162162162162
B_cb = 0.0
EOF
```

**Step 2: Verify TOML syntax**

Run: `python -c "import tomllib; tomllib.load(open('models/pc.toml', 'rb'))"`
Expected: No errors (silent success)

**Step 3: Commit**

```bash
git add models/pc.toml
git commit -m "feat: add PC model in TOML format"
```

---

## Task 4: Add TOML Parsing Tests

**Files:**
- Create: `tests/test_toml_parsing.py`

**Step 1: Write tests for TOML parsing**

```python
"""Tests for TOML config parsing."""

import pytest
from pathlib import Path
from me_sfc.config_model import ConfigModel


def test_toml_valid_config():
    """Valid TOML config loads correctly."""
    config_text = """
    [equations]
    y = "c + g"
    c = "0.8 * y"

    [parameters]
    alpha = 0.8

    [exogenous]
    g = 20.0

    [initial]
    y = 0.0
    c = 0.0
    """
    model = ConfigModel(config_text=config_text)
    assert len(model._var_names) == 2
    assert 'y' in model._var_names
    assert 'c' in model._var_names
    assert model._parameters['alpha'] == 0.8
    assert model._exogenous['g'] == 20.0


def test_toml_missing_equations_section():
    """Error if [equations] section missing."""
    config_text = """
    [parameters]
    alpha = 0.8
    """
    with pytest.raises(ValueError, match="must have \\[equations\\] section"):
        ConfigModel(config_text=config_text)


def test_toml_invalid_syntax():
    """Invalid TOML syntax raises clear error."""
    config_text = """
    [equations
    y = "c + g"
    """
    with pytest.raises(ValueError, match="Invalid TOML syntax"):
        ConfigModel(config_text=config_text)


def test_toml_non_numeric_parameter():
    """Non-numeric parameters rejected."""
    config_text = """
    [equations]
    y = "c + g"

    [parameters]
    alpha = "not a number"
    """
    with pytest.raises(ValueError, match="must be numeric"):
        ConfigModel(config_text=config_text)


def test_toml_sim_file_loads():
    """SIM TOML file loads correctly."""
    model = ConfigModel(config_path="models/sim.toml")
    assert len(model._var_names) == 11
    assert model._parameters['c0'] == 0.6
    assert model._exogenous['g0'] == 20.0


def test_toml_simex_file_loads():
    """SIMEX TOML file loads correctly."""
    model = ConfigModel(config_path="models/simex.toml")
    assert len(model._var_names) == 13
    assert model._parameters['c0'] == 0.6


def test_toml_pc_file_loads():
    """PC TOML file loads correctly."""
    model = ConfigModel(config_path="models/pc.toml")
    assert len(model._var_names) == 10
    assert model._parameters['b0'] == 0.4
    assert model._exogenous['r0'] == 0.025
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_toml_parsing.py -v`
Expected: FAIL with "ModuleNotFoundError" or import errors (TOML parsing not implemented yet)

**Step 3: Commit**

```bash
git add tests/test_toml_parsing.py
git commit -m "test: add TOML parsing tests (failing)"
```

---

## Task 5: Update Parser to Use TOML

**Files:**
- Modify: `src/me_sfc/config_model.py:1-176`

**Step 1: Add tomllib import**

At line 7 (after existing imports), add:

```python
import tomllib
```

**Step 2: Replace _parse_config method**

Replace the entire `_parse_config` method (lines 101-175) with:

```python
def _parse_config(self, config_text: str) -> Dict[str, Any]:
    """Parse TOML config into sections.

    Args:
        config_text: Full config file content (TOML format)

    Returns:
        Dictionary with keys: equations, parameters, exogenous, initial, metadata

    Raises:
        ValueError: If config is malformed or missing required sections
    """
    try:
        config = tomllib.loads(config_text)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Invalid TOML syntax: {e}")

    # Validate required sections
    if 'equations' not in config:
        raise ValueError("Config must have [equations] section")

    sections = {
        'equations': [],
        'parameters': config.get('parameters', {}),
        'exogenous': config.get('exogenous', {}),
        'initial': config.get('initial', {}),
        'metadata': config.get('metadata', {})
    }

    # Convert equations dict to list of "var = expr" strings
    # This preserves the rest of the code that expects this format
    for var, expr in config['equations'].items():
        sections['equations'].append(f"{var} = {expr}")

    # Validate numeric values
    for section_name in ['parameters', 'exogenous', 'initial']:
        for key, value in sections[section_name].items():
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"[{section_name}] {key} must be numeric, got {type(value).__name__}"
                )

    return sections
```

**Step 3: Update __init__ to validate .toml extension**

In `__init__` method, after line 63, replace the file reading section with:

```python
if config_path:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Validate file extension
    if config_path.suffix not in ['.toml', '.txt']:
        raise ValueError(f"Config must be .toml or .txt file, got {config_path.suffix}")

    config_text = config_path.read_text()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_toml_parsing.py -v`
Expected: All tests PASS

**Step 5: Run existing tests to ensure no regression**

Run: `uv run pytest tests/test_plotting.py -v`
Expected: All tests still PASS

**Step 6: Commit**

```bash
git add src/me_sfc/config_model.py
git commit -m "feat: replace custom parser with TOML parser"
```

---

## Task 6: Merge Model Files

**Files:**
- Modify: `src/me_sfc/config_model.py` (will be renamed to model.py)
- Delete: `src/me_sfc/model.py`

**Step 1: Copy methods from model.py to config_model.py**

Read the current model.py and copy these methods (update, simulate, get_results, plot) into config_model.py. They should replace/merge with any existing versions.

Insert after line 301 (end of _equations method):

```python
def get_results(self):
    """Return results as a pandas DataFrame for analysis.

    Converts the list of State namedtuples to a DataFrame (skipping initial zeros).

    Returns:
        pandas.DataFrame with columns for each variable and rows for each time period.
    """
    return pd.DataFrame([s._asdict() for s in self.x[1:]])

def update(self):
    """Update model state by one time period.

    Solves the system of equations using the previous period's solution
    as the initial guess, then appends the new solution to state history.
    """
    initial_guess = self.x[-1]
    solution = fsolve(self._equations, initial_guess)
    # Convert solution to State namedtuple
    self.x.append(self.State(*solution))

def simulate(self, periods):
    """Run the model for multiple periods.

    Args:
        periods: Number of time periods to simulate

    Returns:
        Dictionary of time series for all variables
    """
    for _ in range(periods):
        self.update()
    return self.get_results()

def plot(self, save_path=None, show=True, figsize=None):
    """Plot all variables as time series in separate subplots.

    Args:
        save_path: Optional file path to save figure (e.g., "figures/results.png").
                  Parent directories will be created if they don't exist.
        show: Boolean to display plot interactively (default True)
        figsize: Optional tuple (width, height) for figure size in inches.
                If None, automatically calculated as (15, 3 * rows)

    Raises:
        ValueError: If results are empty or contains no time periods
    """
    # Get results from the model (must be DataFrame)
    results = self.get_results()

    if not isinstance(results, pd.DataFrame):
        raise TypeError("get_results() must return a pandas DataFrame")

    if results.empty:
        raise ValueError("No results to plot. Run simulate() first.")

    # Calculate grid dimensions
    n_vars = len(results.columns)
    cols = min(3, n_vars)  # Max 3 columns
    rows = ceil(n_vars / cols)

    # Set figure size
    if figsize is None:
        figsize = (15, 3 * rows)

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Handle single subplot case (axes is not an array)
    if n_vars == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    # Plot each variable
    for idx, var_name in enumerate(results.columns):
        row = idx // cols
        col = idx % cols
        ax = axes[row][col]

        # Plot time series
        ax.plot(results.index, results[var_name], linewidth=2)
        ax.set_title(var_name, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add axis labels only on edge subplots
        if row == rows - 1:  # Bottom row
            ax.set_xlabel("Period", fontsize=10)
        if col == 0:  # Left column
            ax.set_ylabel("Value", fontsize=10)

    # Hide unused subplots
    for idx in range(n_vars, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row][col].axis("off")

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save if path provided
    if save_path:
        # Create parent directories if needed
        parent_dir = os.path.dirname(save_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)
```

**Step 2: Add required imports**

At top of config_model.py, add these imports (merge with existing):

```python
import os
from math import ceil
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd
```

**Step 3: Rename class from ConfigModel to Model**

Replace `class ConfigModel(Model):` with `class Model:` (remove inheritance)

Update docstring to remove reference to inheritance.

**Step 4: Update class __repr__**

Change line 337 from:
```python
return f"ConfigModel({len(self._var_names)} vars, {len(self._sections['equations'])} eqs)"
```

to:
```python
return f"Model({len(self._var_names)} vars, {len(self._sections['equations'])} eqs)"
```

**Step 5: Delete old model.py and rename config_model.py**

```bash
git rm src/me_sfc/model.py
git mv src/me_sfc/config_model.py src/me_sfc/model.py
```

**Step 6: Update imports in __init__.py**

Edit `src/me_sfc/__init__.py`:

```python
"""SFC macroeconomic modeling framework."""

from me_sfc.model import Model

__all__ = ['Model']
```

**Step 7: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

**Step 8: Commit**

```bash
git add src/me_sfc/
git commit -m "refactor: merge Model and ConfigModel into single class"
```

---

## Task 7: Update Examples to Use TOML

**Files:**
- Modify: `examples/run_config_models.py:1-174`

**Step 1: Update import statement**

Change line 4 from:
```python
from me_sfc.config_model import ConfigModel
```

to:
```python
from me_sfc.model import Model
```

**Step 2: Update model instantiations**

Replace all occurrences of `ConfigModel` with `Model`.

Update file paths from `.txt` to `.toml`:
- Line ~20: `"models/sim.txt"` → `"models/sim.toml"`
- Line ~60: `"models/simex.txt"` → `"models/simex.toml"`
- Line ~100: `"models/pc.txt"` → `"models/pc.toml"`

**Step 3: Run examples to verify they work**

Run: `uv run python examples/run_config_models.py`
Expected: Script completes without errors, generates 3 PNG files

**Step 4: Verify output files exist**

Run: `ls -la figures/*.png`
Expected: sim_config.png, simex_config.png, pc_config.png all present

**Step 5: Commit**

```bash
git add examples/run_config_models.py
git commit -m "refactor: update examples to use TOML configs"
```

---

## Task 8: Update Test Imports

**Files:**
- Modify: `tests/test_plotting.py:1-50`
- Modify: `tests/test_toml_parsing.py:1-100`

**Step 1: Update test_plotting.py imports**

Change line 4 from:
```python
from me_sfc.model import Model
```

to (if it was importing ConfigModel):
```python
from me_sfc.model import Model
```

This might already be correct - verify the import.

**Step 2: Update test_toml_parsing.py imports**

Change line 4 from:
```python
from me_sfc.config_model import ConfigModel
```

to:
```python
from me_sfc.model import Model
```

Replace all occurrences of `ConfigModel` with `Model` in the test file.

**Step 3: Run all tests**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/
git commit -m "refactor: update test imports for merged Model class"
```

---

## Task 9: Update Documentation

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Step 1: Update CLAUDE.md**

Replace config format examples with TOML format. Find section showing config file format and update:

```markdown
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
```

Update usage examples to use `.toml` extension:

```markdown
### Quick Start

```python
from me_sfc.model import Model

# Load and run a model
model = Model(config_path="models/sim.toml")
results = model.simulate(periods=100)
model.plot()
```

### Available Config Models

- `models/sim.toml`: Simplest model with government money (11 equations)
- `models/simex.toml`: SIM with expectations (13 equations)
- `models/pc.toml`: Portfolio choice model (10 equations)
```

**Step 2: Update README.md**

Similarly update README.md with TOML format examples and `.toml` file extensions.

Find any references to:
- `ConfigModel` → change to `Model`
- `.txt` config files → change to `.toml`
- Config format examples → update to TOML syntax

**Step 3: Verify documentation clarity**

Read through both files to ensure consistency and clarity.

**Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: update documentation for TOML format"
```

---

## Task 10: Remove Legacy Files

**Files:**
- Delete: `models/sim.txt`
- Delete: `models/simex.txt`
- Delete: `models/pc.txt`
- Delete: `DESIGN_ANALYSIS.md`

**Step 1: Remove legacy .txt config files**

```bash
git rm models/sim.txt models/simex.txt models/pc.txt
```

**Step 2: Remove superseded design doc**

```bash
git rm DESIGN_ANALYSIS.md
```

**Step 3: Verify no references to deleted files**

Run: `grep -r "\.txt" . --include="*.py" --include="*.md" | grep -v ".gitignore" | grep -v ".worktrees"`
Expected: No references to sim.txt, simex.txt, or pc.txt in code/docs

**Step 4: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git commit -m "chore: remove legacy .txt configs and superseded design doc"
```

---

## Task 11: Add Regression Tests

**Files:**
- Create: `tests/test_toml_regression.py`

**Step 1: Write regression tests comparing TOML output**

```python
"""Regression tests to verify TOML models produce correct results."""

import pytest
import numpy as np
from me_sfc.model import Model


def test_sim_toml_converges():
    """SIM model converges to expected steady state."""
    model = Model(config_path="models/sim.toml")
    results = model.simulate(periods=100)

    # Check convergence (last value close to steady state)
    final_Y = results['Y'].iloc[-1]
    assert 99.0 < final_Y < 101.0, f"Expected Y≈100, got {final_Y}"

    final_H = results['H_h'].iloc[-1]
    assert 79.0 < final_H < 81.0, f"Expected H≈80, got {final_H}"


def test_simex_toml_converges():
    """SIMEX model converges to expected steady state."""
    model = Model(config_path="models/simex.toml")
    results = model.simulate(periods=100)

    final_Y = results['Y'].iloc[-1]
    assert 99.0 < final_Y < 101.0, f"Expected Y≈100, got {final_Y}"


def test_pc_toml_stable():
    """PC model maintains steady state from proper initial values."""
    model = Model(config_path="models/pc.toml")
    results = model.simulate(periods=50)

    # PC starts at steady state, should remain stable
    initial_Y = results['Y'].iloc[0]
    final_Y = results['Y'].iloc[-1]

    # Allow small drift but should be stable
    assert np.abs(final_Y - initial_Y) < 0.1, f"Y drifted from {initial_Y} to {final_Y}"


def test_toml_models_have_correct_variable_counts():
    """Each model has expected number of variables."""
    sim = Model(config_path="models/sim.toml")
    assert len(sim._var_names) == 11, "SIM should have 11 variables"

    simex = Model(config_path="models/simex.toml")
    assert len(simex._var_names) == 13, "SIMEX should have 13 variables"

    pc = Model(config_path="models/pc.toml")
    assert len(pc._var_names) == 10, "PC should have 10 variables"
```

**Step 2: Run regression tests**

Run: `uv run pytest tests/test_toml_regression.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add tests/test_toml_regression.py
git commit -m "test: add regression tests for TOML models"
```

---

## Task 12: Final Verification

**Files:**
- All files (verification step)

**Step 1: Run complete test suite**

Run: `uv run pytest tests/ -v`
Expected: All tests PASS

**Step 2: Run examples**

Run: `uv run python examples/run_config_models.py`
Expected: Completes successfully, generates figures

**Step 3: Verify figures generated**

Run: `ls -la figures/`
Expected: sim_config.png, simex_config.png, pc_config.png present

**Step 4: Check git status**

Run: `git status`
Expected: Working tree clean (all changes committed)

**Step 5: View commit history**

Run: `git log --oneline -12`
Expected: See all commits from this plan

**Step 6: Create summary commit if needed**

If any final cleanup needed, commit it:

```bash
git add .
git commit -m "chore: final cleanup for TOML migration"
```

---

## Success Criteria

- ✅ All three models (SIM, SIMEX, PC) converted to TOML format
- ✅ Custom parser replaced with tomllib (115 lines → ~30 lines)
- ✅ Model and ConfigModel merged into single Model class
- ✅ All tests pass (plotting, parsing, regression)
- ✅ Examples run successfully with TOML configs
- ✅ Documentation updated (CLAUDE.md, README.md)
- ✅ Legacy .txt files removed
- ✅ Clean git history with descriptive commits

## Testing Commands

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_toml_parsing.py -v

# Run examples
uv run python examples/run_config_models.py

# Verify TOML syntax
python -c "import tomllib; print(tomllib.load(open('models/sim.toml', 'rb')))"
```

## Rollback Plan

If issues arise:

```bash
# Return to previous commit
git log --oneline
git reset --hard <commit-hash>

# Or revert specific commit
git revert <commit-hash>
```

---

**Plan saved to:** `docs/plans/2025-11-08-toml-migration.md`
