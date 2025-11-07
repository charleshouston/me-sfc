# Config-Based Model Design Analysis

## Objective
Refactor the SFC model framework from class-per-model to config-file-per-model with a single universal model class.

## Current Structure Analysis
- Base `Model` class with abstract `_equations()` method
- Each model (SIM, SIMEX, PC) is a separate Python class
- Equations hardcoded as Python expressions
- Parameters passed to `__init__`
- State managed via namedtuples
- Built-in simulation and plotting

## Approach Options

### Approach 1: Plain Text DSL with eval()
**Config format**: Plain text like example
```
y = cons + g
yd = y - t + r(-1)*b_h(-1)
```

**Implementation**:
- Regex to parse variable assignments
- Convert `var(-1)` to `prev['var']`
- Use Python `eval()` with restricted namespace

**Evaluation**:
- ✅ Simple to implement (DRY: reuse Python's parser)
- ✅ Readable config format
- ✅ Flexible expressions (operators, functions)
- ⚠️ Security concerns with eval()
- ⚠️ Runtime errors harder to debug
- **YAGNI Score**: 9/10 - Minimal complexity

### Approach 2: Sympy-based Symbolic
**Config format**: Plain text or YAML
```
equations:
  - y = cons + g
  - yd = y - t + r_lag1*b_h_lag1
```

**Implementation**:
- Parse into sympy expressions
- Symbolic variable creation
- Numerical evaluation via lambdify

**Evaluation**:
- ✅ Mathematical rigor
- ✅ Can compute Jacobians
- ❌ Overkill for numeric models (YAGNI violation)
- ❌ Performance overhead
- ❌ More complex syntax requirements
- **YAGNI Score**: 4/10 - Too much machinery

### Approach 3: YAML + Custom AST Parser
**Config format**: Structured YAML
```yaml
equations:
  y:
    type: identity
    value: "cons + g"
  yd:
    type: behavioral
    value: "y - t + lag(r, 1)*lag(b_h, 1)"
```

**Implementation**:
- Custom recursive descent parser
- Build expression tree
- Interpret tree at runtime

**Evaluation**:
- ✅ Very safe (no eval)
- ✅ Structured validation
- ❌ Most complex to implement (DRY violation - reinventing parser)
- ❌ Maintenance burden
- **YAGNI Score**: 3/10 - Over-engineered

### Approach 4: Python Module Config
**Config format**: Python file
```python
def equations(current, prev, params):
    return {
        'y': current['cons'] + params['g'],
        'yd': current['y'] - current['t'] + prev['r'] * prev['b_h']
    }
```

**Implementation**:
- Import config as module
- Execute function to get equations
- Direct Python execution

**Evaluation**:
- ✅ No parsing needed (DRY)
- ✅ IDE support, type hints
- ✅ Full Python flexibility
- ⚠️ Requires Python knowledge
- ⚠️ Less declarative than text
- **YAGNI Score**: 7/10 - Practical but less user-friendly

### Approach 5: Hybrid Safe Evaluator (RECOMMENDED)
**Config format**: Enhanced plain text
```
###### Equations
y = cons + g
yd = y - t + r(-1)*b_h(-1)

###### Parameters
alpha1 = 0.6
alpha2 = 0.4

###### Exogenous
g = 20

###### Initial
y = 100
```

**Implementation**:
- Parse config into sections (equations, params, exogenous, initial)
- Convert `var(-1)` → `prev['var']` via regex
- Use `eval()` with controlled namespace (`__builtins__={}`)
- Add safe math functions (basic operators only)

**Evaluation**:
- ✅ Simple, readable config
- ✅ Balance of safety and simplicity
- ✅ Follows DRY (reuse Python evaluation)
- ✅ Easy to validate and debug
- ✅ User-friendly for economists
- **YAGNI Score**: 9/10 - Just right

### Approach 6: NumExpr Evaluation
**Config format**: Plain text
```
y = cons + g
yd = y - t + r_prev * b_h_prev
```

**Implementation**:
- Preprocess to replace lags with `_prev` suffix
- Use `numexpr.evaluate()` for fast math
- Build variable context

**Evaluation**:
- ✅ Fast numeric evaluation
- ✅ Safer than pure eval
- ⚠️ Limited function support
- ⚠️ May not handle all patterns
- **YAGNI Score**: 6/10 - Performance not critical yet

## Selected Approach: Hybrid Safe Evaluator (Approach 5)

### Rationale
1. **DRY Principle**: Reuses Python's expression evaluator rather than building a parser
2. **YAGNI Principle**: Provides exactly what's needed without over-engineering
3. **User Experience**: Simple text format familiar to economists
4. **Safety**: Restricted namespace prevents malicious code
5. **Maintainability**: Minimal code to maintain, clear structure
6. **Extensibility**: Easy to add validation, equation metadata later

### Implementation Plan
1. Create `ConfigModel` class inheriting from `Model`
2. Implement config parser for sections: equations, parameters, exogenous, initial
3. Convert lag notation `var(-1)` to dictionary access
4. Build safe evaluation namespace
5. Generate `_equations()` method dynamically
6. Preserve existing Model features (plot, simulate, get_results)
7. Create config files for SIM, SIMEX, PC

### Key Design Decisions
- **Config format**: Plain text with `######` section markers (readable, git-friendly)
- **Lag notation**: Keep `var(-1)` syntax (familiar to economists)
- **Equation order**: Parser determines dependency order (no manual sorting needed)
- **Validation**: Check variable names, detect circular dependencies
- **Error messages**: Show equation text when solver fails

## Next Steps
1. Implement prototype to validate approach
2. Test with simplified SIM model
3. Extend to full SIM, SIMEX, PC models
4. Update documentation and examples
