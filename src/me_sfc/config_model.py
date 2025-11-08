"""Config-based model system for SFC macroeconomic models.

This module implements a universal model class that reads model specifications
from simple text configuration files rather than requiring separate Python classes.
"""

import re
import numpy as np
import tomllib
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Tuple, Any
from me_sfc.model import Model


class ConfigModel(Model):
    """Universal SFC model that reads specifications from config files.

    This class allows economists to define models using simple text files
    with equations, parameters, and initial values, without writing Python code.

    Example config file format:
        ###### Equations
        y = c + g
        yd = y - t
        c = c0 * yd + c1 * h(-1)

        ###### Parameters
        c0 = 0.6
        c1 = 0.4

        ###### Exogenous
        g = 20

        ###### Initial
        h = 0

    Features:
    - Simple text format with section markers (######)
    - Lag notation: var(-1) for previous period values
    - Automatic variable detection and State namedtuple creation
    - Safe evaluation with restricted namespace
    - All base Model features: simulate(), plot(), get_results()
    """

    def __init__(self, config_path: str = None, config_text: str = None):
        """Initialize model from config file or text.

        Args:
            config_path: Path to config file (relative or absolute)
            config_text: Config as string (alternative to file)

        Raises:
            ValueError: If neither or both arguments provided
            FileNotFoundError: If config_path doesn't exist
            ValueError: If config is malformed
        """
        if (config_path is None) == (config_text is None):
            raise ValueError("Provide exactly one of config_path or config_text")

        if config_path:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            # Validate file extension
            if config_path.suffix != '.toml':
                raise ValueError(f"Config must be .toml file, got {config_path.suffix}")

            config_text = config_path.read_text()

        # Parse configuration
        self._config_text = config_text
        self._sections = self._parse_config(config_text)

        # Extract variable names from equations
        self._var_names = self._extract_variables(self._sections['equations'])

        # Create State namedtuple dynamically
        self.State = namedtuple('State', self._var_names)

        # Store parameters and exogenous as instance attributes
        self._parameters = self._sections['parameters']
        self._exogenous = self._sections['exogenous']

        # Build safe evaluation namespace
        self._namespace = self._build_namespace()

        # Initialize state history with initial values
        initial_values = self._sections['initial']

        # Fill in zeros for any variables not specified in initial section
        for var in self._var_names:
            if var not in initial_values:
                initial_values[var] = 0.0

        # Create initial state (in correct order)
        initial_state = self.State(**{var: initial_values[var] for var in self._var_names})
        self.x = [initial_state]

        # Preprocess equations (convert lags)
        self._processed_equations = [
            self._convert_lags(eq, self._var_names)
            for eq in self._sections['equations']
        ]

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

    def _extract_variables(self, equations: List[str]) -> List[str]:
        """Extract variable names from left-hand side of equations.

        Args:
            equations: List of equation strings like "y = c + g"

        Returns:
            List of variable names in order of appearance

        Raises:
            ValueError: If equation format is invalid
        """
        var_names = []
        for eq in equations:
            if '=' not in eq:
                raise ValueError(f"Equation must contain '=': {eq}")

            lhs = eq.split('=')[0].strip()

            # Validate variable name
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', lhs):
                raise ValueError(f"Invalid variable name on LHS: {lhs}")

            var_names.append(lhs)

        return var_names

    def _convert_lags(self, equation: str, var_names: List[str]) -> str:
        """Convert lag notation var(-1) to prev['var'].

        Args:
            equation: Equation string like "c = c0 * yd + c1 * h(-1)"
            var_names: List of valid variable names

        Returns:
            Converted equation like "c = c0 * yd + c1 * prev['h']"
        """
        result = equation

        # Sort by length descending to match longer names first
        for var in sorted(var_names, key=len, reverse=True):
            # Match var(-1) or var( -1 ) with optional spaces
            pattern = rf'\b{re.escape(var)}\s*\(\s*-1\s*\)'
            replacement = f"prev['{var}']"
            result = re.sub(pattern, replacement, result)

        return result

    def _build_namespace(self) -> Dict[str, Any]:
        """Build safe namespace for equation evaluation.

        Returns:
            Dictionary with safe built-ins, parameters, and exogenous values
        """
        # Start with empty builtins for security
        namespace = {
            '__builtins__': {},
            # Add safe math functions
            'abs': abs,
            'min': min,
            'max': max,
            'pow': pow,
            'round': round,
            # Add numpy if needed
            'exp': np.exp,
            'log': np.log,
            'sqrt': np.sqrt,
        }

        # Add parameters and exogenous values
        namespace.update(self._parameters)
        namespace.update(self._exogenous)

        return namespace

    def _equations(self, x):
        """Define the system of equations for this model.

        This implements the abstract method from Model base class.

        Args:
            x: Current guess for solution vector (numpy array)

        Returns:
            List of residuals that should equal zero at equilibrium
        """
        # Convert solution vector to State namedtuple
        current = self.State(*x)

        # Get previous period state
        prev_state = self.x[-1]

        # Build evaluation namespace for this period
        ns = self._namespace.copy()

        # Add current values to namespace
        for var in self._var_names:
            ns[var] = getattr(current, var)

        # Add previous values as dictionary
        prev = {var: getattr(prev_state, var) for var in self._var_names}
        ns['prev'] = prev

        # Evaluate each equation and compute residual
        residuals = []

        for eq in self._processed_equations:
            try:
                # Split into LHS = RHS
                lhs, rhs = eq.split('=', 1)
                lhs = lhs.strip()
                rhs = rhs.strip()

                # Evaluate both sides
                lhs_val = eval(lhs, ns)
                rhs_val = eval(rhs, ns)

                # Residual: LHS - RHS should equal 0
                residual = lhs_val - rhs_val
                residuals.append(residual)

            except Exception as e:
                raise RuntimeError(f"Error evaluating equation '{eq}': {e}")

        return residuals

    def get_config_info(self) -> str:
        """Return formatted summary of model configuration.

        Returns:
            Multi-line string describing the model
        """
        info = []
        info.append("=" * 60)
        info.append("CONFIG-BASED SFC MODEL")
        info.append("=" * 60)
        info.append(f"\nVariables ({len(self._var_names)}): {', '.join(self._var_names)}")
        info.append(f"\nEquations ({len(self._sections['equations'])}):")
        for i, eq in enumerate(self._sections['equations'], 1):
            info.append(f"  {i}. {eq}")

        if self._parameters:
            info.append(f"\nParameters:")
            for name, value in self._parameters.items():
                info.append(f"  {name} = {value}")

        if self._exogenous:
            info.append(f"\nExogenous:")
            for name, value in self._exogenous.items():
                info.append(f"  {name} = {value}")

        if self._sections['initial']:
            info.append(f"\nInitial Values:")
            for name, value in self._sections['initial'].items():
                info.append(f"  {name} = {value}")

        info.append("=" * 60)
        return '\n'.join(info)

    def __repr__(self):
        """Return string representation of model."""
        return f"ConfigModel({len(self._var_names)} vars, {len(self._sections['equations'])} eqs)"
