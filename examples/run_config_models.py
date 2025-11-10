"""Example: Running SFC models from config files.

This demonstrates how to use the ConfigModel class to load and simulate
models defined in simple text configuration files.
"""

from me_sfc.model import Model
from pathlib import Path


def run_sim_model():
    """Run the SIM model from config file."""
    print("\n" + "=" * 70)
    print("RUNNING SIM MODEL (from config)")
    print("=" * 70)

    # Load model from config file
    model = Model(config_path="models/sim.toml")

    # Display model info
    print(model.get_config_info())

    # Simulate
    print("\nSimulating 100 periods...")
    results = model.simulate(periods=100)

    # Display results
    print("\nFinal Period Results (t=100):")
    print("-" * 45)
    final = results.iloc[-1]
    print(f"National Income (Y):       {final['Y']:>10.2f}")
    print(f"Disposable Income (YD):    {final['YD']:>10.2f}")
    print(f"Consumption (C_d):         {final['C_d']:>10.2f}")
    print(f"Government Spending (G_s): {final['G_s']:>10.2f}")
    print(f"Taxes (T_d):               {final['T_d']:>10.2f}")
    print(f"Household Wealth (H_h):    {final['H_h']:>10.2f}")

    # Check convergence
    y_change = abs(results["Y"].iloc[-1] - results["Y"].iloc[-2])
    print(f"\nConvergence: ΔY = {y_change:.6f}")

    # Save plot
    model.plot(save_path="figures/sim_config.png", show=False)
    print("\nPlot saved to: figures/sim_config.png")

    return model, results


def run_simex_model():
    """Run the SIMEX model from config file."""
    print("\n" + "=" * 70)
    print("RUNNING SIMEX MODEL (from config)")
    print("=" * 70)

    # Load model
    model = Model(config_path="models/simex.toml")

    print(f"\n{model}")
    print(f"Variables: {', '.join(model._var_names)}")

    # Simulate
    print("\nSimulating 100 periods...")
    results = model.simulate(periods=100)

    # Display results
    print("\nFinal Period Results (t=100):")
    print("-" * 45)
    final = results.iloc[-1]
    print(f"National Income (Y):           {final['Y']:>10.2f}")
    print(f"Disposable Income (YD):        {final['YD']:>10.2f}")
    print(f"Expected Income (YDe):         {final['YDe']:>10.2f}")
    print(f"Consumption (C_d):             {final['C_d']:>10.2f}")
    print(f"Household Wealth (H_h):        {final['H_h']:>10.2f}")
    print(f"Household Money Demand (H_d):  {final['H_d']:>10.2f}")

    # Check convergence
    y_change = abs(results["Y"].iloc[-1] - results["Y"].iloc[-2])
    print(f"\nConvergence: ΔY = {y_change:.6f}")

    # Save plot
    model.plot(save_path="figures/simex_config.png", show=False)
    print("\nPlot saved to: figures/simex_config.png")

    return model, results


def run_pc_model():
    """Run the PC model from config file."""
    print("\n" + "=" * 70)
    print("RUNNING PC MODEL (from config)")
    print("=" * 70)

    # Load model
    model = Model(config_path="models/pc.toml")

    print(f"\n{model}")
    print(f"Variables: {', '.join(model._var_names)}")

    # Simulate
    print("\nSimulating 100 periods...")
    results = model.simulate(periods=100)

    # Display results
    print("\nFinal Period Results (t=100):")
    print("-" * 45)
    final = results.iloc[-1]
    for var in ['Y', 'YD', 'C', 'T', 'V', 'H_h', 'B_h', 'B_s', 'B_cb', 'H_s']:
        print(f"{var:>6}: {final[var]:>12.4f}")

    # Check convergence
    y_change = abs(results["Y"].iloc[-1] - results["Y"].iloc[-2])
    print(f"\nConvergence: ΔY = {y_change:.6f}")

    # Save plot
    model.plot(save_path="figures/pc_config.png", show=False)
    print("\nPlot saved to: figures/pc_config.png")

    return model, results


def compare_models():
    """Compare steady-state values across models."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON: Steady-State Values")
    print("=" * 70)

    # Run all models
    sim_model, sim_results = Model(config_path="models/sim.toml"), None
    simex_model, simex_results = Model(config_path="models/simex.toml"), None
    pc_model, pc_results = Model(config_path="models/pc.toml"), None

    sim_results = sim_model.simulate(periods=100)
    simex_results = simex_model.simulate(periods=100)
    pc_results = pc_model.simulate(periods=100)

    print(f"\n{'Variable':<10} {'SIM':>12} {'SIMEX':>12} {'PC':>12}")
    print("-" * 50)

    # Compare common variables
    for var in ['Y', 'C_d' if 'C_d' in sim_results else 'C']:
        sim_val = sim_results[var].iloc[-1] if var in sim_results else None
        simex_val = simex_results[var].iloc[-1] if var in simex_results else None
        pc_val = pc_results['C'].iloc[-1] if var == 'C_d' and 'C' in pc_results else \
                 pc_results[var].iloc[-1] if var in pc_results else None

        print(f"{var:<10} {sim_val:>12.2f} {simex_val:>12.2f} {pc_val:>12.2f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  CONFIG-BASED SFC MODEL EXAMPLES".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)

    # Run each model
    run_sim_model()
    run_simex_model()
    run_pc_model()

    # Compare models
    compare_models()

    print("\n" + "=" * 70)
    print("ALL MODELS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - figures/sim_config.png")
    print("  - figures/simex_config.png")
    print("  - figures/pc_config.png")
    print("\n")
