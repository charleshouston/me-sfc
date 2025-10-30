# Model SIM
# Simplest Model with Government Money

# The model has 11 equations, 11 unknowns, and 3 exogenous variables
# Y, YD, T_d, T_s, H_s, H_h, G_s, C_s, C_d, N_s, N_d
# Exogenous variables: G_d, theta, W

# Whatever is demanded (services, taxes, and labour) is always supplied
# within the period:
# C_s = C_d
# G_s = G_d
# T_s = T_d
# N_s = N_d

# Diposable income:
# YD = W * N_s - T_s

# Taxes:
# T_d = theta * W * N_s, where theta < 1

# Consumption:
# C_d = c0 * YD + c1 * H_h[-1]

# Budget constraint of government:
# \Delta H_s = H_s - H_s[-1] = G_d - T_d
# H_s = H_s[-1] + G_d - T_d

# Budget constraint of households:
# \Delta H_h = H_h - H_h[-1] = YD - C_d
# H_h = H_h[-1] + YD - C_d

# National income identity
# Y = C_s + G_s
# Y = W * N_d
# N_d = Y / W

import numpy as np
from scipy.optimize import fsolve


class SIM:
    def __init__(self, c0, c1, theta, g0, W):
        # Exogenous constants
        self.c0 = c0  # Consumption from disposable income
        self.c1 = c1  # Wealth effect
        self.theta = theta  # Tax rate
        self.g0 = g0  # Government spending
        self.W = W  # Wage rate

        self.x = [np.zeros(11)]  # solution vector

    def update(self):
        """Update for one time step."""

        # Define the system of equations
        def f(x):
            Y, YD, T_d, T_s, H_s, H_h, G_s, C_s, C_d, N_s, N_d = x

            H_h_prev = self.x[-1][5]
            H_s_prev = self.x[-1][4]

            eq1 = C_s - C_d
            eq2 = G_s - self.g0
            eq3 = T_s - T_d
            eq4 = N_s - N_d
            eq5 = YD - (self.W * N_s - T_s)
            eq6 = T_d - (self.theta * self.W * N_s)
            eq7 = C_d - (self.c0 * YD + self.c1 * H_h_prev)
            eq8 = H_s - (H_s_prev + self.g0 - T_d)
            eq9 = H_h - (H_h_prev + YD - C_d)
            eq10 = Y - (C_s + G_s)
            eq11 = N_d - (Y / self.W)

            return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11]

        initial_guess = self.x[-1]

        Y = fsolve(f, initial_guess)
        self.x.append(Y)

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

    def get_results(self):
        """Return results as a dictionary of time series.

        Returns:
            Dictionary mapping variable names to numpy arrays of their values over time
        """
        arr = np.array(self.x[1:])  # Skip initial zeros
        return {
            'Y': arr[:, 0],    # National income
            'YD': arr[:, 1],   # Disposable income
            'T_d': arr[:, 2],  # Tax demand
            'T_s': arr[:, 3],  # Tax supply
            'H_s': arr[:, 4],  # Government money supply
            'H_h': arr[:, 5],  # Household money holdings
            'G_s': arr[:, 6],  # Government spending supply
            'C_s': arr[:, 7],  # Consumption supply
            'C_d': arr[:, 8],  # Consumption demand
            'N_s': arr[:, 9],  # Employment supply
            'N_d': arr[:, 10]  # Employment demand
        }


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
    print(f"National Income (Y):       {results['Y'][-1]:>10.2f}")
    print(f"Disposable Income (YD):    {results['YD'][-1]:>10.2f}")
    print(f"Consumption (C):           {results['C_d'][-1]:>10.2f}")
    print(f"Government Spending (G):   {results['G_s'][-1]:>10.2f}")
    print(f"Taxes (T):                 {results['T_d'][-1]:>10.2f}")
    print(f"Household Wealth (H_h):    {results['H_h'][-1]:>10.2f}")
    print(f"Government Debt (H_s):     {results['H_s'][-1]:>10.2f}")
    print(f"Employment (N):            {results['N_d'][-1]:>10.2f}")
    print()

    # Check if model has reached steady state
    y_change = abs(results['Y'][-1] - results['Y'][-2])
    if y_change < 0.01:
        print(f"Model has converged to steady state (ΔY = {y_change:.4f})")
    else:
        print(f"Model still adjusting (ΔY = {y_change:.4f})")
