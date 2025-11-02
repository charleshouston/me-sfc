# Model SIM
# Simplest Model with Government Money

import numpy as np
from collections import namedtuple
from me_sfc.model import Model


class SIM(Model):
    # Define state structure as a namedtuple for readable, fast access
    State = namedtuple(
        "State",
        [
            "Y",  # National income
            "YD",  # Disposable income
            "T_d",  # Tax demand
            "T_s",  # Tax supply
            "H_s",  # Government money supply
            "H_h",  # Household money holdings
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
        self.x = [self.State(*np.zeros(11))]

    def _equations(self, x):
        """System of 11 equations for the SIM model.

        Args:
            x: Solution vector (array) for current period values

        Returns:
            List of 11 residuals that should equal zero at equilibrium
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

        # Eq 5: Disposable income is wages minus taxes
        eq5 = current.YD - (self.W * current.N_s - current.T_s)

        # Eq 6: Taxes demanded are a proportion of wages
        eq6 = current.T_d - (self.theta * self.W * current.N_s)

        # Eq 7: Consumption function with wealth effect
        eq7 = current.C_d - (self.c0 * current.YD + self.c1 * prev.H_h)

        # Eq 8: Government budget constraint
        eq8 = current.H_s - (prev.H_s + self.g0 - current.T_d)

        # Eq 9: Household budget constraint
        eq9 = current.H_h - (prev.H_h + current.YD - current.C_d)

        # Eqs 10-11: National income identity
        eq10 = current.Y - (current.C_s + current.G_s)
        eq11 = current.N_d - (current.Y / self.W)

        return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11]


if __name__ == "__main__":
    # Standard calibration from Godley & Lavoie
    # c0: marginal propensity to consume out of income
    # c1: propensity to consume out of wealth
    # theta: tax rate
    # g0: government spending
    # W: wage rate
    model = SIM(c0=0.6, c1=0.4, theta=0.2, g0=20, W=1)

    # Run simulation for 100 periods
    results = model.simulate(periods=100)

    # Display final period results
    print("SIM Model - Final Period Results (t=100)")
    print("=" * 45)
    print(f"National Income (Y):       {results['Y'].iloc[-1]:>10.2f}")
    print(f"Disposable Income (YD):    {results['YD'].iloc[-1]:>10.2f}")
    print(f"Consumption (C):           {results['C_d'].iloc[-1]:>10.2f}")
    print(f"Government Spending (G):   {results['G_s'].iloc[-1]:>10.2f}")
    print(f"Taxes (T):                 {results['T_d'].iloc[-1]:>10.2f}")
    print(f"Household Wealth (H_h):    {results['H_h'].iloc[-1]:>10.2f}")
    print(f"Government Debt (H_s):     {results['H_s'].iloc[-1]:>10.2f}")
    print(f"Employment (N):            {results['N_d'].iloc[-1]:>10.2f}")
    print()

    # Check if model has reached steady state
    y_change = abs(results["Y"].iloc[-1] - results["Y"].iloc[-2])
    if y_change < 0.01:
        print(f"Model has converged to steady state (ΔY = {y_change:.4f})")
    else:
        print(f"Model still adjusting (ΔY = {y_change:.4f})")

    # Demonstrate plotting functionality
    print("\nGenerating plots...")
    print("Close the plot window to continue.")

    # Display plots interactively
    model.plot()

    # Also save to file
    model.plot(save_path="figures/sim_baseline.png", show=False)
    print("Plots saved to figures/sim_baseline.png")
