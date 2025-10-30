"""Base class for Stock-Flow Consistent macroeconomic models."""

from abc import ABC, abstractmethod
import os
from math import ceil
import matplotlib.pyplot as plt


class Model(ABC):
    """Abstract base class for SFC models.

    All SFC models should inherit from this class and implement:
    - get_results(): Return dictionary of time series data

    The base class provides:
    - plot(): Visualize results with automatic subplot layout
    - simulate(): Default implementation (can be overridden)
    """

    @abstractmethod
    def get_results(self):
        """Return results as a dictionary of time series.

        Returns:
            Dictionary mapping variable names to numpy arrays of their values over time.
            Example: {'Y': array([...]), 'C': array([...]), ...}
        """
        pass

    @abstractmethod
    def update(self):
        """Update model state by one time period.

        This method should:
        - Solve the system of equations for the current period
        - Update internal state with new values
        """
        pass

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
            ValueError: If results dictionary is empty or contains no time periods
        """
        # Get results from the model
        results = self.get_results()

        # Validate results
        if not results:
            raise ValueError("No results to plot. Run simulate() first.")

        # Check if any data exists
        first_var = next(iter(results.values()))
        if len(first_var) == 0:
            raise ValueError("Results contain no time periods. Run simulate() first.")

        # Calculate grid dimensions
        n_vars = len(results)
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
        for idx, (var_name, values) in enumerate(results.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row][col]

            # Plot time series
            periods = range(len(values))
            ax.plot(periods, values, linewidth=2)
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
