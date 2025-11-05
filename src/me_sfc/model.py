"""Base class for Stock-Flow Consistent macroeconomic models."""

from abc import ABC, abstractmethod
import os
from math import ceil
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd


class Model(ABC):
    """Abstract base class for SFC models.

    All SFC models should inherit from this class and implement:
    - _equations(x): Define the system of equations as residuals

    Subclasses must define:
    - State: Class-level namedtuple defining the model's state structure
    - self.x: List initialized with State namedtuples

    Example:
        class YourModel(Model):
            State = namedtuple('State', ['Y', 'C', 'I', ...])

            def __init__(self, ...):
                self.x = [self.State(*np.zeros(n))]

            def _equations(self, x):
                current = self.State(*x)
                prev = self.x[-1]
                # Use current.Y, prev.C, etc.

    The base class provides:
    - update(): Solves equations and converts solution to State namedtuple
    - get_results(): Converts State history to pandas DataFrame
    - plot(): Visualize results with automatic subplot layout
    - simulate(): Default implementation (can be overridden)
    """

    @abstractmethod
    def _equations(self, x):
        """Define the system of equations for this model.

        Args:
            x: Current guess for solution vector (numpy array or list)

        Returns:
            List of residuals that should equal zero at equilibrium.
            Each equation should be expressed as: expression - target = 0
        """
        pass

    def get_results(self):
        """Return results as a pandas DataFrame for analysis.

        Converts the list of State namedtuples to a DataFrame (skipping initial zeros).

        Returns:
            pandas.DataFrame with columns for each variable and rows for each time period.

        Example:
            DataFrame with columns ['Y', 'C', 'I', ...] where each row is a time period
        """
        return pd.DataFrame([s._asdict() for s in self.x[1:]])

    def update(self):
        """Update model state by one time period.

        Solves the system of equations using the previous period's solution
        as the initial guess, then appends the new solution to state history.

        Assumes subclass defines self.State namedtuple for the hybrid pattern.
        """
        initial_guess = self.x[-1]
        solution = fsolve(self._equations, initial_guess)
        # Convert solution to State namedtuple
        self.x.append(self.State(*solution))

    def simulate(self, periods):
        """Run the model for multiple periods.

        Default implementation calls update() for each period.
        Subclasses can override this method for custom simulation logic.

        Args:
            periods: Number of time periods to simulate

        Returns:
            Dictionary of time series for all variables
        """
        for _ in range(periods):
            self.update()
        return self.get_results()

    def plot(self, save_path=None, show=True, figsize=None):
        """Plot all variables as time series in separate subplots.

        Args:
            save_path: Optional file path to save figure (e.g., "figures/results.png").
                      Parent directories will be created if they don't exist.
            show: Boolean to display plot interactively (default True)
            figsize: Optional tuple (width, height) for figure size in inches.
                    If None, automatically calculated as (15, 3 * rows)

        Raises:
            ValueError: If results are empty or contains no time periods
        """
        # Get results from the model (must be DataFrame)
        results = self.get_results()

        if not isinstance(results, pd.DataFrame):
            raise TypeError("get_results() must return a pandas DataFrame")

        if results.empty:
            raise ValueError("No results to plot. Run simulate() first.")

        # Calculate grid dimensions
        n_vars = len(results.columns)
        cols = min(3, n_vars)  # Max 3 columns
        rows = ceil(n_vars / cols)

        # Set figure size
        if figsize is None:
            figsize = (15, 3 * rows)

        # Create subplots
        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        # Handle single subplot case (axes is not an array)
        if n_vars == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        # Plot each variable
        for idx, var_name in enumerate(results.columns):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col]

            # Plot time series
            ax.plot(results.index, results[var_name], linewidth=2)
            ax.set_title(var_name, fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)

            # Add axis labels only on edge subplots
            if row == rows - 1:  # Bottom row
                ax.set_xlabel("Period", fontsize=10)
            if col == 0:  # Left column
                ax.set_ylabel("Value", fontsize=10)

        # Hide unused subplots
        for idx in range(n_vars, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row][col].axis("off")

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save if path provided
        if save_path:
            # Create parent directories if needed
            parent_dir = os.path.dirname(save_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Figure saved to: {save_path}")

        # Show if requested
        if show:
            plt.show()
        else:
            plt.close(fig)

    def _balance_sheet_structure(self, state):
        """Define the balance sheet structure for this model.

        Subclasses should override this method to specify their balance sheet.
        This is an optional method - models without balance sheets need not implement it.

        Args:
            state: State namedtuple for a specific period

        Returns:
            dict with two keys:
                'assets': List of asset type names (e.g., ["Money (H)", "Bills (B)"])
                'sectors': Dict mapping sector names to lists of values
                           (e.g., {"Households": [H_h, B_h], "Government": [-B_s]})

        Example:
            return {
                'assets': ["Money (H)"],
                'sectors': {
                    "Households": [state.H_h],
                    "Government": [-state.H_s]
                }
            }
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented balance sheet structure. "
            "Override _balance_sheet_structure(state) to enable this feature."
        )

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
            NotImplementedError: If subclass has not defined balance sheet structure.
        """
        # Check if simulation has been run
        if len(self.x) <= 1:
            raise ValueError(
                "No simulation results available. Run simulate() first."
            )

        # Get state at specified period
        # x[0] is initial state, x[1] is period 0, x[2] is period 1, etc.
        state = self.x[period] if period == -1 else self.x[period + 1]

        # Get model-specific structure
        structure = self._balance_sheet_structure(state)

        # Build DataFrame from structure
        df = pd.DataFrame(structure['sectors'], index=structure['assets'])

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
