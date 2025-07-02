import os
import matplotlib.pyplot as plt
from typing import Any

from warprec.recommenders.base_recommender import Recommender

PLOT_DIR = "custom_plots/evaluation_callback_plots"


def on_model_evaluation(model: Recommender, results: dict, **kwargs: Any):
    """Code snippet example of callback usage.

    Args:
        model (Recommender): The trained recommender model instance.
        results (dict): A dictionary containing the evaluation results.
        **kwargs (Any): Other arguments passed to the callback.
    """
    print("\n--- STARTING CUSTOM USER ANALYSIS ---")
    print(f"Current Model: {model.name}")

    # Extract data for plotting
    # Results are organized by 'Test' and 'Validation', then by k-value.
    # Each k-value has a dictionary of metrics.

    k_values = []
    ndcg_test_values = []
    ndcg_validation_values = []  # Assuming you have similar validation results

    # Process 'Test' results
    if "Test" in results and results["Test"]:
        # Extract and sort k-values (important for plotting order)
        sorted_k_test = sorted([int(k) for k in results["Test"].keys()])

        for k in sorted_k_test:
            k_values.append(k)
            # Check if the metric exists for that k-value
            if "nDCG" in results["Test"][k]:
                ndcg_test_values.append(results["Test"][k]["nDCG"])
            else:
                ndcg_test_values.append(0.0)  # Default value if metric is missing

    # Process 'Validation' results if present
    validation_exists = "Validation" in results and results["Validation"]
    if validation_exists:
        # For validation, assume the same k-values.
        # If validation k-values could differ, they should be handled separately.
        for k in sorted_k_test:  # Use the same k_values to align plots
            if "nDCG" in results["Validation"][k]:
                ndcg_validation_values.append(results["Validation"][k]["nDCG"])
            else:
                ndcg_validation_values.append(0.0)
    else:
        print("No 'Validation' results found for nDCG metric.")

    # --- Creating the nDCG vs k Plot ---
    if k_values:  # Plot only if there's data
        plt.figure(figsize=(10, 6))  # Figure size for better visualization

        # Plot for Test results
        plt.plot(
            k_values,
            ndcg_test_values,
            marker="o",
            linestyle="-",
            color="blue",
            label="nDCG (Test)",
        )

        # Plot for Validation results (if present)
        if validation_exists:
            plt.plot(
                k_values,
                ndcg_validation_values,
                marker="x",
                linestyle="--",
                color="red",
                label="nDCG (Validation)",
            )

        plt.title(f"nDCG vs. k for Model: {model.name}")
        plt.xlabel("Top-K")
        plt.ylabel("nDCG")
        plt.xticks(k_values)  # Display only the k-values we have
        plt.grid(True, linestyle="--", alpha=0.7)  # Add a grid for readability
        plt.legend()  # Show the legend to distinguish lines
        plt.tight_layout()  # Adjust plot parameters for a tight layout

        # Save the plot
        # Create a 'custom_plots' directory if it doesn't exist
        os.makedirs(PLOT_DIR, exist_ok=True)

        # Filename based on the model's name
        plot_filename = os.path.join(
            PLOT_DIR, f"ndcg_vs_k_{model.name.replace(' ', '_')}.png"
        )
        plt.savefig(plot_filename)
        print(f"nDCG vs k plot saved as: {plot_filename}")

        plt.close()  # Close the figure to free up memory
    else:
        print("No valid data to generate the nDCG plot.")

    print("--- END CUSTOM USER ANALYSIS ---")
