"""Tests for TOML config parsing."""

import pytest
from pathlib import Path
from me_sfc.model import Model


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
    model = Model(config_text=config_text)
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
        Model(config_text=config_text)


def test_toml_invalid_syntax():
    """Invalid TOML syntax raises clear error."""
    config_text = """
    [equations
    y = "c + g"
    """
    with pytest.raises(ValueError, match="Invalid TOML syntax"):
        Model(config_text=config_text)


def test_toml_non_numeric_parameter():
    """Non-numeric parameters rejected."""
    config_text = """
    [equations]
    y = "c + g"

    [parameters]
    alpha = "not a number"
    """
    with pytest.raises(ValueError, match="must be numeric"):
        Model(config_text=config_text)


def test_toml_sim_file_loads():
    """SIM TOML file loads correctly."""
    model = Model(config_path="models/sim.toml")
    assert len(model._var_names) == 11
    assert model._parameters['c0'] == 0.6
    assert model._exogenous['g0'] == 20.0


def test_toml_simex_file_loads():
    """SIMEX TOML file loads correctly."""
    model = Model(config_path="models/simex.toml")
    assert len(model._var_names) == 13
    assert model._parameters['c0'] == 0.6


def test_toml_pc_file_loads():
    """PC TOML file loads correctly."""
    model = Model(config_path="models/pc.toml")
    assert len(model._var_names) == 10
    assert model._parameters['b0'] == 0.4
    assert model._exogenous['r0'] == 0.025
