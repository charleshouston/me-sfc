"""Tests for balance sheet matrix functionality in SIMEX model."""

import pytest
import numpy as np
import pandas as pd
from me_sfc.simex import SIMEX


class TestSIMEXBalanceSheet:
    """Test suite for SIMEX model balance sheet matrix generation."""

    @pytest.fixture
    def simulated_model(self):
        """Create a SIMEX model and simulate it."""
        model = SIMEX(c0=0.6, c1=0.4, theta=0.2, g0=20, W=1)
        model.simulate(periods=10)
        return model

    def test_returns_dataframe(self, simulated_model):
        """Balance sheet should return a pandas DataFrame."""
        bs = simulated_model.get_balance_sheet()
        assert isinstance(bs, pd.DataFrame)

    def test_has_correct_sectors(self, simulated_model):
        """Balance sheet should have Households, Government, and SUM columns."""
        bs = simulated_model.get_balance_sheet()
        expected_sectors = ["Households", "Government", "SUM"]
        assert list(bs.columns) == expected_sectors

    def test_has_correct_assets(self, simulated_model):
        """Balance sheet should have row for Money only (SIMEX has no bills)."""
        bs = simulated_model.get_balance_sheet()
        expected_assets = ["Money (H)"]
        assert list(bs.index) == expected_assets

    def test_rows_sum_to_zero(self, simulated_model):
        """Each row should sum to approximately zero (accounting identity)."""
        bs = simulated_model.get_balance_sheet()
        # Check that SUM column is all close to zero
        for asset in bs.index:
            assert abs(bs.loc[asset, "SUM"]) < 1e-6, f"{asset} does not sum to zero"

    def test_money_consistency(self, simulated_model):
        """Money held by households should equal money supplied by government."""
        bs = simulated_model.get_balance_sheet()
        # H_h (households hold) + (-H_s) (government supplies) = 0
        assert abs(bs.loc["Money (H)", "SUM"]) < 1e-6

    def test_default_period_is_last(self, simulated_model):
        """Default period should be the last simulated period."""
        bs_default = simulated_model.get_balance_sheet()
        bs_last = simulated_model.get_balance_sheet(period=-1)
        pd.testing.assert_frame_equal(bs_default, bs_last)

    def test_specific_period(self, simulated_model):
        """Should be able to get balance sheet for specific period."""
        bs_period_5 = simulated_model.get_balance_sheet(period=5)
        # Values should differ from last period (model evolves over time)
        bs_last = simulated_model.get_balance_sheet(period=-1)
        # At least one value should be different
        assert not bs_period_5.equals(bs_last)

    def test_raises_error_if_no_simulation(self):
        """Should raise error if trying to get balance sheet before simulation."""
        model = SIMEX(c0=0.6, c1=0.4, theta=0.2, g0=20, W=1)
        # Model only has initial state (x[0]), no simulation results
        with pytest.raises(ValueError, match="No simulation results"):
            model.get_balance_sheet()

    def test_values_match_state(self, simulated_model):
        """Balance sheet values should match the model's state at that period."""
        period = 5
        bs = simulated_model.get_balance_sheet(period=period)
        # Access state at period (remember x[0] is initial, x[1] is first period)
        state = simulated_model.x[period + 1]  # +1 because x[0] is initial

        # Check households money (asset)
        assert bs.loc["Money (H)", "Households"] == state.H_h
        # Check government money supply (liability, so negative)
        assert bs.loc["Money (H)", "Government"] == -state.H_s

    def test_money_stocks_are_positive(self, simulated_model):
        """Money stocks should be positive after simulation."""
        bs = simulated_model.get_balance_sheet()
        # Households hold positive money
        assert bs.loc["Money (H)", "Households"] > 0
        # Government liability is negative of that amount
        assert bs.loc["Money (H)", "Government"] < 0
