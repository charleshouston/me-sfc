"""Tests for Model base class plotting functionality."""

import numpy as np
from model import Model


class DummyModel(Model):
    """Minimal model for testing plotting functionality."""

    def __init__(self):
        self.simulated = False

    def update(self):
        """Dummy update method."""
        self.simulated = True

    def get_results(self):
        """Return dummy results for testing."""
        if not self.simulated:
            return {}
        return {
            'Variable_A': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            'Variable_B': np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
            'Variable_C': np.array([2.5, 2.5, 2.5, 2.5, 2.5])
        }


def test_plot_requires_results():
    """Test that plot raises error when no results available."""
    model = DummyModel()

    try:
        model.plot(show=False)
        assert False, "Expected ValueError for empty results"
    except ValueError as e:
        assert "No results to plot" in str(e)

    print("✓ Test passed: plot() raises error for empty results")


def test_plot_saves_file():
    """Test that plot saves file when path provided."""
    import os
    import tempfile

    model = DummyModel()
    model.simulated = True

    # Create temporary file path
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Remove the file (we just wanted the path)
        os.unlink(tmp_path)

        # Generate plot
        model.plot(save_path=tmp_path, show=False)

        # Check file was created
        assert os.path.exists(tmp_path), "Plot file was not created"
        assert os.path.getsize(tmp_path) > 0, "Plot file is empty"

        print(f"✓ Test passed: plot() saves file successfully ({os.path.getsize(tmp_path)} bytes)")
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_plot_grid_layout():
    """Test that plot handles different numbers of variables."""
    model = DummyModel()
    model.simulated = True

    # Test with 3 variables (should work without errors)
    try:
        model.plot(show=False)
        print("✓ Test passed: plot() handles 3 variables correctly")
    except Exception as e:
        assert False, f"plot() failed with 3 variables: {e}"


if __name__ == "__main__":
    print("Running plotting tests...\n")
    test_plot_requires_results()
    test_plot_saves_file()
    test_plot_grid_layout()
    print("\nAll tests passed!")
