"""Base class for Stock-Flow Consistent macroeconomic models."""

from abc import ABC, abstractmethod


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
