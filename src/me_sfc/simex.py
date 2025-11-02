# Model SIMEX
# Model SIM with Expectations

import numpy as np
from collections import namedtuple
from me_sfc.model import Model


class SIMEX(Model):
    # State
    State = namedtuple(
        "State",
        [
            "Y",  # National income
            "YD",  # Disposable income
            "YDe",  # Expected disposable income
            "T_d",  # Tax demand
            "T_s",  # Tax supply
            "H_s",  # Government money supply
            "H_h",  # Household money holdings
            "H_d",  # Household money demand
            "G_s",  # Government spending supply
            "C_s",  # Consumption supply
            "C_d",  # Consumption demand
            "N_s",  # Employment supply
            "N_d",  # Employment demand
        ],
    )

    def __init__(self, c0, c1, theta, g0, W):
        # Exogenous constants
        self.c0 = c0  # Consumption from disposable income
        self.c1 = c1  # Wealth effect
        self.theta = theta  # Tax rate
        self.g0 = g0  # Government spending
        self.W = W  # Wage rate

        # Initialize state history with named tuples for readable access
        self.x = [self.State(*np.zeros(13))]

    def _equations(self, x):
        """System of 13 equations for the SIMEX model.

        Args:
            x: Solution vector (array) for current period values

        Returns:
            List of 13 residuals that should equal zero at equilibrium
        """
        # Unpack current period values (being solved)
        current = self.State(*x)

        # Get previous period values (from history) - no more magic indices!
        prev = self.x[-1]

        # Eqs 1-4: Whatever is demanded is supplied in the period
        eq1 = current.C_s - current.C_d
        eq2 = current.G_s - self.g0
        eq3 = current.T_s - current.T_d
        eq4 = current.N_s - current.N_d

        # Eq 5: Expected disposable income is last period's disposable income
        eq5 = current.YDe - prev.YD

        # Eq 6: Disposable income is wages minus taxes
        eq6 = current.YD - (self.W * current.N_s - current.T_s)

        # Eq 7: Taxes demanded are a proportion of wages
        eq7 = current.T_d - (self.theta * self.W * current.N_s)

        # Eq 8: Consumption function with wealth effect and expected income
        eq8 = current.C_d - (self.c0 * current.YDe + self.c1 * prev.H_h)

        # Eq 9: Government budget constraint
        eq9 = current.H_s - (prev.H_s + self.g0 - current.T_d)

        # Eq 10: Household budget constraint
        eq10 = current.H_h - (prev.H_h + current.YD - current.C_d)

        # Eq 11-12: National income identity
        eq11 = current.Y - (current.C_s + current.G_s)
        eq12 = current.N_d - (current.Y / self.W)

        # Eq 13: Household money demand
        eq13 = current.H_d - (prev.H_h + current.YDe - current.C_d)

        return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11, eq12, eq13]


if __name__ == "__main__":
    # Standard calibration
    model = SIMEX(c0=0.6, c1=0.4, theta=0.2, g0=20, W=1)

    results = model.simulate(periods=100)

    # Display final period results
    print("SIMEX Model - Final Period Results (t=100)")
    print("=" * 45)
    print(f"National Income (Y):       {results['Y'].iloc[-1]:>10.2f}")
    print(f"Disposable Income (YD):    {results['YD'].iloc[-1]:>10.2f}")
    print(f"Expected Disposable Income (YDe): {results['YDe'].iloc[-1]:>10.2f}")
    print(f"Consumption (C):          {results['C_d'].iloc[-1]:>10.2f}")
    print(f"Government Spending (G):      {results['G_s'].iloc[-1]:>10.2f}")
    print(f"Taxes (T):              {results['T_d'].iloc[-1]:>10.2f}")
    print(f"Household Wealth (H_h):       {results['H_h'].iloc[-1]:>10.2f}")
    print(f"Household Money Demand (H_d): {results['H_d'].iloc[-1]:>10.2f}")
    print(f"Government Debt (H_s):        {results['H_s'].iloc[-1]:>10.2f}  ")
    print(f"Employment (N):           {results['N_d'].iloc[-1]:>10.2f}")
    print()

    # Check if the model reached a steady state
    y_change = abs(results["Y"].iloc[-1] - results["Y"].iloc[-2])
    if y_change < 0.01:
        print(f"Model has converged to steady state (ΔY = {y_change:.4f})")
    else:
        print(f"Model still adjusting (ΔY = {y_change:.4f})")

    model.plot(save_path="figures/simex_baseline.png", show=False)
    print("Plot saved to figures/simex_baseline.png")
