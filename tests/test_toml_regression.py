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
    """PC model maintains steady state after convergence."""
    model = Model(config_path="models/pc.toml")
    results = model.simulate(periods=100)

    # Check stability over last 50 periods after convergence
    mid_Y = results['Y'].iloc[50]
    final_Y = results['Y'].iloc[-1]

    # Should be stable after initial convergence
    assert np.abs(final_Y - mid_Y) < 0.1, f"Y drifted from {mid_Y} to {final_Y}"


def test_toml_models_have_correct_variable_counts():
    """Each model has expected number of variables."""
    sim = Model(config_path="models/sim.toml")
    assert len(sim._var_names) == 11, "SIM should have 11 variables"

    simex = Model(config_path="models/simex.toml")
    assert len(simex._var_names) == 13, "SIMEX should have 13 variables"

    pc = Model(config_path="models/pc.toml")
    assert len(pc._var_names) == 10, "PC should have 10 variables"
