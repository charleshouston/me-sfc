"""Tests for balance sheet matrix functionality."""

import pytest
import numpy as np
import pandas as pd
from me_sfc.pc import PC


class TestBalanceSheet:
    """Test suite for balance sheet matrix generation."""

    @pytest.fixture
    def simulated_model(self):
        """Create a PC model and simulate it."""
        model = PC(c0=0.6, c1=0.4, b0=0.4, b1=0.2, b2=0.1, theta=0.2, r0=0.025, g0=20)
        model.simulate(periods=10)
        return model

    def test_returns_dataframe(self, simulated_model):
        """Balance sheet should return a pandas DataFrame."""
        bs = simulated_model.get_balance_sheet()
        assert isinstance(bs, pd.DataFrame)

    def test_has_correct_sectors(self, simulated_model):
        """Balance sheet should have Households, Government, Central Bank, and SUM columns."""
        bs = simulated_model.get_balance_sheet()
        expected_sectors = ["Households", "Government", "Central Bank", "SUM"]
        assert list(bs.columns) == expected_sectors

    def test_has_correct_assets(self, simulated_model):
        """Balance sheet should have rows for Money and Bills."""
        bs = simulated_model.get_balance_sheet()
        expected_assets = ["Money (H)", "Bills (B)"]
        assert list(bs.index) == expected_assets

    def test_rows_sum_to_zero(self, simulated_model):
        """Each row should sum to approximately zero (accounting identity)."""
        bs = simulated_model.get_balance_sheet()
        # Check that SUM column is all close to zero
        for asset in bs.index:
            assert abs(bs.loc[asset, "SUM"]) < 1e-6, f"{asset} does not sum to zero"

    def test_money_consistency(self, simulated_model):
        """Money held by households should equal money supplied by central bank."""
        bs = simulated_model.get_balance_sheet()
        # H_h (households hold) + (-H_s) (central bank supplies) = 0
        assert abs(bs.loc["Money (H)", "SUM"]) < 1e-6

    def test_bills_consistency(self, simulated_model):
        """Bills held by households and CB should equal bills supplied by government."""
        bs = simulated_model.get_balance_sheet()
        # B_h + B_cb - B_s = 0
        assert abs(bs.loc["Bills (B)", "SUM"]) < 1e-6

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
        model = PC(c0=0.6, c1=0.4, b0=0.4, b1=0.2, b2=0.1, theta=0.2, r0=0.025, g0=20)
        # Model only has initial state (x[0]), no simulation results
        with pytest.raises(ValueError, match="No simulation results"):
            model.get_balance_sheet()

    def test_values_match_state(self, simulated_model):
        """Balance sheet values should match the model's state at that period."""
        period = 5
        bs = simulated_model.get_balance_sheet(period=period)
        # Access state at period (remember x[0] is initial, x[1] is first period)
        state = simulated_model.x[period + 1]  # +1 because x[0] is initial

        # Check households money
        assert bs.loc["Money (H)", "Households"] == state.H_h
        # Check central bank money supply (negative because it's a liability)
        assert bs.loc["Money (H)", "Central Bank"] == -state.H_s
        # Check households bills
        assert bs.loc["Bills (B)", "Households"] == state.B_h
        # Check central bank bills
        assert bs.loc["Bills (B)", "Central Bank"] == state.B_cb
        # Check government bills (negative because it's a liability)
        assert bs.loc["Bills (B)", "Government"] == -state.B_s
