"""Prototype for config-based model approach.

Tests the Hybrid Safe Evaluator approach before full implementation.
"""

import re
import numpy as np
from collections import namedtuple


def parse_config(config_text):
    """Parse config file into sections."""
    sections = {
        'equations': [],
        'parameters': {},
        'exogenous': {},
        'initial': {}
    }

    current_section = None

    for line in config_text.strip().split('\n'):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith('##'):
            # Check for section markers
            if 'equation' in line.lower():
                current_section = 'equations'
            elif 'parameter' in line.lower():
                current_section = 'parameters'
            elif 'exogenous' in line.lower():
                current_section = 'exogenous'
            elif 'initial' in line.lower() or 'starting' in line.lower():
                current_section = 'initial'
            continue

        if current_section == 'equations':
            # Parse: var = expression
            if '=' in line:
                sections['equations'].append(line)
        elif current_section in ['parameters', 'exogenous', 'initial']:
            # Parse: var = value
            if '=' in line:
                var, value = line.split('=', 1)
                var = var.strip()
                value = float(value.strip())
                sections[current_section][var] = value

    return sections


def convert_lags(equation_text, variable_names):
    """Convert lag notation var(-1) to prev['var'].

    Args:
        equation_text: Equation string like "y = c + g(-1)"
        variable_names: List of valid variable names to replace

    Returns:
        Modified equation with prev['var'] replacements
    """
    result = equation_text

    # Sort by length descending to match longer names first (e.g., y_d before y)
    for var in sorted(variable_names, key=len, reverse=True):
        # Match var(-1) pattern
        pattern = rf'\b{re.escape(var)}\s*\(\s*-1\s*\)'
        replacement = f"prev['{var}']"
        result = re.sub(pattern, replacement, result)

    return result


def test_prototype():
    """Test the prototype with a simplified SIM model."""

    # Simplified SIM config (just a few equations)
    config = """
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
    yd = 0
    c = 0
    h = 0
    t = 0
    """

    print("=" * 60)
    print("PROTOTYPE TEST: Config-Based Model Parser")
    print("=" * 60)

    # Parse config
    sections = parse_config(config)

    print("\n1. PARSED SECTIONS:")
    print(f"   Equations: {len(sections['equations'])} found")
    for eq in sections['equations']:
        print(f"      {eq}")
    print(f"   Parameters: {sections['parameters']}")
    print(f"   Exogenous: {sections['exogenous']}")
    print(f"   Initial: {sections['initial']}")

    # Extract variable names from equations (left side of =)
    var_names = []
    for eq in sections['equations']:
        var = eq.split('=')[0].strip()
        var_names.append(var)

    print(f"\n2. DETECTED VARIABLES: {var_names}")

    # Convert lag notation
    print("\n3. LAG CONVERSION TEST:")
    for eq in sections['equations']:
        converted = convert_lags(eq, var_names)
        if converted != eq:
            print(f"   Original:  {eq}")
            print(f"   Converted: {converted}")

    # Test equation evaluation
    print("\n4. EQUATION EVALUATION TEST:")

    # Build safe namespace
    safe_namespace = {
        '__builtins__': {},
        'abs': abs,
        'min': min,
        'max': max,
        'pow': pow,
    }

    # Add parameters and exogenous
    safe_namespace.update(sections['parameters'])
    safe_namespace.update(sections['exogenous'])

    # Create initial state
    State = namedtuple('State', var_names)
    initial_state = State(**sections['initial'])

    print(f"   Initial state: {initial_state}")

    # Simulate solving one period
    def equations(x, prev_state):
        """Return residuals for equation system."""
        current = State(*x)

        # Build namespace for evaluation
        ns = safe_namespace.copy()

        # Add current values
        for var in var_names:
            ns[var] = getattr(current, var)

        # Add previous values
        prev = {}
        for var in var_names:
            prev[var] = getattr(prev_state, var)
        ns['prev'] = prev

        residuals = []
        for eq in sections['equations']:
            # Convert lags
            eq_converted = convert_lags(eq, var_names)

            # Split into LHS = RHS
            lhs, rhs = eq_converted.split('=')
            lhs = lhs.strip()
            rhs = rhs.strip()

            # Evaluate: lhs - rhs should equal 0
            lhs_val = eval(lhs, ns)
            rhs_val = eval(rhs, ns)
            residual = lhs_val - rhs_val
            residuals.append(residual)

        return residuals

    # Test with non-zero guess
    test_guess = np.array([80.0, 64.0, 60.0, 50.0, 16.0])  # y, yd, c, h, t
    test_prev = State(*[0.0] * len(var_names))  # Start from zeros

    try:
        residuals = equations(test_guess, test_prev)
        print(f"   Test guess: {test_guess}")
        print(f"   Residuals: {np.array(residuals)}")
        print(f"   ✅ Equation evaluation successful!")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    # Try to solve
    print("\n5. SOLVER TEST:")
    from scipy.optimize import fsolve

    try:
        solution = fsolve(lambda x: equations(x, test_prev), test_guess)
        print(f"   Solution: {solution}")
        print(f"   Residuals: {equations(solution, test_prev)}")

        # Create state
        solved_state = State(*solution)
        print(f"\n   Solved state:")
        for var in var_names:
            print(f"      {var}: {getattr(solved_state, var):.4f}")

        print(f"\n   ✅ Solver successful!")
    except Exception as e:
        print(f"   ❌ Solver error: {e}")
        return False

    print("\n" + "=" * 60)
    print("PROTOTYPE TEST: ✅ PASSED")
    print("Hybrid Safe Evaluator approach is viable!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_prototype()
    exit(0 if success else 1)
