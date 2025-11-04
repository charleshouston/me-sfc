# Model PC
# Agents make a portfolio choice between money and other financial assets


from me_sfc.model import Model
from collections import namedtuple
import numpy as np
import pandas as pd


class PC(Model):
    State = namedtuple(
        "State",
        [
            "Y",  # National income
            "YD",  # Disposable income
            "T",  # Taxes
            "C",  # Consumption
            "V",  # Household wealth
            "H_h",  # Household demand for cash
            "B_h",  # Household demand for bills
            "H_s",  # Supply of cash by central bank
            "B_cb",  # Demand for bills by central bank
            "B_s",  # Supply of bills by government
        ],
    )

    def __init__(self, c0, c1, b0, b1, b2, theta, r0, g0, initial_state=None):
        self.c0 = c0  # Consumption from disposable income
        self.c1 = c1  # Consumption from wealth
        self.b0 = b0  # Base demand for bills
        self.b1 = b1  # Sensitivity of bill demand to interest rate
        self.b2 = b2  # Sensitivity of bill demand to transactions (disposable income)
        self.theta = theta  # Tax rate
        self.r0 = r0  # Interest rate on bills
        self.g0 = g0  # Government spending

        # PC model requires non-zero initial wealth to avoid division by zero
        if initial_state is None:
            # Reasonable initial guess based on steady-state approximations
            # In steady state: Y ≈ g0/(1 - c0*(1-theta))
            Y_init = g0 / (1 - c0 * (1 - theta))
            YD_init = Y_init * (1 - theta)
            T_init = Y_init * theta
            C_init = c0 * YD_init

            # Wealth accumulates until saving stops: V such that c1*V = (1-c0)*YD
            V_init = ((1 - c0) / c1) * YD_init

            # Portfolio allocation: B_h = V*(b0 + b1*r) - b2*YD
            B_h_init = V_init * (b0 + b1 * r0) - b2 * YD_init
            H_h_init = V_init - B_h_init

            # Government starts with enough bills issued to cover deficit history
            B_s_init = (
                B_h_init + 5
            )  # Government issues slightly more than household demand

            initial_state = self.State(
                Y=Y_init,
                YD=YD_init,
                T=T_init,
                C=C_init,
                V=V_init,
                H_h=H_h_init,
                B_h=B_h_init,
                H_s=H_h_init,
                B_cb=B_s_init - B_h_init,  # Central bank holds the residual
                B_s=B_s_init,
            )

        self.x = [initial_state]

    def _equations(self, x):
        current = self.State(*x)

        prev = self.x[-1]

        # Eq 1: National income
        eq1 = current.Y - (current.C + self.g0)

        # Eq 2: Disposable income is income less taxes plus interest on bills
        eq2 = current.YD - (current.Y - current.T + self.r0 * prev.B_h)

        # Eq 3: Taxes are a proportion of income plus interest on bills
        eq3 = current.T - self.theta * (current.Y + self.r0 * prev.B_h)

        # Eq 4: Change in household wealth is disposable income less consumption
        eq4 = current.V - (prev.V + current.YD - current.C)

        # Eq 5: Consumption function
        eq5 = current.C - (self.c0 * current.YD + self.c1 * prev.V)

        # Eq 6: Demand for bills is positively related to interest rate and negatively to disposable income (transactions)
        eq6 = current.B_h - (
            current.V * (self.b0 + self.b1 * self.r0) - self.b2 * current.YD
        )

        # Eq 7: Demand for cash is the residual
        eq7 = current.H_h - (current.V - current.B_h)

        # Eq 8: Government budget constraint
        # Change in bills equals spending (gov consumption plus interest transfers) less revenues (taxes and interest on central bank bills)
        eq8 = (
            current.B_s
            - prev.B_s
            - ((self.g0 + self.r0 * prev.B_s) - (current.T + self.r0 * prev.B_cb))
        )

        # Eq 9: Capital account of central bank
        # Additions to stock of high-powered money equal to additions to demand for bills by the central bank
        eq9 = current.H_s - prev.H_s - (current.B_cb - prev.B_cb)

        # Eq 10: Demand for bills by central bank
        # Central bank is residual purchaser of bills
        eq10 = current.B_cb - (current.B_s - current.B_h)

        return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10])

    def get_balance_sheet(self, period=-1):
        """Generate the balance sheet matrix for a specific period.

        The balance sheet matrix shows assets (+) and liabilities (-) for each sector.
        Each row (asset type) must sum to zero to satisfy accounting consistency.

        Args:
            period: Time period to display (default -1 for last period).
                   Period 0 is first simulated period (x[1] in state history).

        Returns:
            pandas.DataFrame with sectors as columns and asset types as rows.
            Includes a SUM column to verify accounting identity (should be ~0).

        Raises:
            ValueError: If no simulation results exist (only initial state).
        """
        # Check if simulation has been run
        if len(self.x) <= 1:
            raise ValueError(
                "No simulation results available. Run simulate() first."
            )

        # Get state at specified period
        # x[0] is initial state, x[1] is period 0, x[2] is period 1, etc.
        state = self.x[period] if period == -1 else self.x[period + 1]

        # Build balance sheet matrix
        # Rows: asset types, Columns: sectors
        data = {
            "Households": [
                state.H_h,  # Money held by households (asset)
                state.B_h,  # Bills held by households (asset)
            ],
            "Government": [
                0.0,  # Government doesn't hold money
                -state.B_s,  # Bills issued by government (liability)
            ],
            "Central Bank": [
                -state.H_s,  # Money supplied by CB (liability)
                state.B_cb,  # Bills held by CB (asset)
            ],
        }

        index = ["Money (H)", "Bills (B)"]
        df = pd.DataFrame(data, index=index)

        # Add SUM column to verify accounting identity
        df["SUM"] = df.sum(axis=1)

        return df

    def print_balance_sheet(self, period=-1):
        """Print the balance sheet matrix in a readable format.

        Args:
            period: Time period to display (default -1 for last period).
        """
        bs = self.get_balance_sheet(period=period)

        # Determine which period we're showing
        if period == -1:
            period_label = len(self.x) - 2  # -1 for initial state, -1 for 0-indexing
        else:
            period_label = period

        print(f"\nBalance Sheet Matrix - Period {period_label}")
        print("=" * 70)
        print(bs.to_string(float_format=lambda x: f"{x:>12.4f}"))
        print("=" * 70)
        print("Note: Assets are positive (+), Liabilities are negative (-)")
        print("      Each row should sum to approximately zero.\n")


if __name__ == "__main__":
    # Standard calibration (adjusted to ensure positive bill holdings)
    # b0 increased to 0.4 and b2 reduced to 0.1 to ensure B_h > 0
    model = PC(c0=0.6, c1=0.4, b0=0.4, b1=0.2, b2=0.1, theta=0.2, r0=0.025, g0=20)
    results = model.simulate(periods=100)

    # Display final period results
    print("PC Model - Final Period Results (t=100)")
    print("=" * 45)
    final_state = results.iloc[-1]
    for var in final_state.index:
        print(f"{var:>6}: {final_state[var]:>10.4f}")
    print()

    # Check convergence
    y_change = abs(results["Y"].iloc[-1] - results["Y"].iloc[-2])
    if y_change < 0.01:
        print(f"Model has converged to steady state (ΔY = {y_change:.4f})")
    else:
        print(f"Model still adjusting (ΔY = {y_change:.4f})")

    model.plot(save_path="figures/pc_baseline.png", show=False)
    print("Plot saved to figures/pc_baseline.png")
