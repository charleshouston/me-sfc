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
    - get_results(): Return pandas DataFrame of time series data

    Recommended pattern (for performance + readability):
    - Store state as list of namedtuples during simulation (fast)
    - Convert to DataFrame in get_results() for analysis (convenient)
    - Example:
        State = namedtuple('State', ['Y', 'C', 'I', ...])
        self.x = [State(*np.zeros(n))]  # Initialize

    Subclasses must initialize:
    - self.x: List of solution vectors (namedtuples or arrays)

    The base class provides:
    - update(): Solves equations and updates state
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

    @abstractmethod
    def get_results(self):
        """Return results as a pandas DataFrame for analysis.

        Returns:
            pandas.DataFrame with columns for each variable and rows for each time period.

        Example:
            DataFrame with columns ['Y', 'C', 'I', ...] where each row is a time period
        """
        pass

    def update(self):
        """Update model state by one time period.

        Solves the system of equations using the previous period's solution
        as the initial guess, then appends the new solution to state history.
        """
        initial_guess = self.x[-1]
        solution = fsolve(self._equations, initial_guess)
        self.x.append(solution)

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
            ax.set_title(var_name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add axis labels only on edge subplots
            if row == rows - 1:  # Bottom row
                ax.set_xlabel('Period', fontsize=10)
            if col == 0:  # Left column
                ax.set_ylabel('Value', fontsize=10)

        # Hide unused subplots
        for idx in range(n_vars, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row][col].axis('off')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Save if path provided
        if save_path:
            # Create parent directories if needed
            parent_dir = os.path.dirname(save_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        # Show if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
