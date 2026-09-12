import os
import sys

from maxent_solver import (
    read_input, optimize_weights, predict_probabilities,
    write_output, compute_accuracy, compute_errors, has_hidden_structure,
)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass


# Configuration
INPUT_FILE = "sample_input_files/input_with_hidden_structure_variable.txt"
OUTPUT_FILE = "output_files/predictions.txt"

# Hyperparameters
SIGMA = 1000.0        # L2 prior std dev (higher = weaker prior).
                      # 1000 ~ no regularization.
ALPHA = 1.0           # L1 prior strength (used when REG == 'L1')
REG = 'L2'            # 'L1', 'L2', or None
MU = 0.0              # Prior mean
INIT_WEIGHTS = 0.0    # Initial weight value
LOWER_BOUND = 0.0     # Non-negative weights


# Main
def run_model(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    """Fit a model to one file and write its prediction tableau."""
    print("=" * 60)
    print("MaxEntLab")
    print("=" * 60)

    data = read_input(input_file)
    inputs = data['Input'].unique()
    hidden = has_hidden_structure(data)
    result = optimize_weights(
        data, reg=REG, sigma=SIGMA, alpha=ALPHA, mu=MU,
        init_weights=INIT_WEIGHTS, lower_bound=LOWER_BOUND,
    )
    con_names = result['constraint_names']

    print(f"Loaded: {len(inputs)} tableaux, "
          f"{len(con_names)} constraints, "
          f"hidden structure: {'yes' if hidden else 'no'}")

    print("\nWeights:")
    for name, w in zip(con_names, result['weights']):
        print(f"  {name}: {w:.4f}")

    # Write the prediction tableau
    preds = predict_probabilities(data, result['weights'], con_names)

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    write_output(preds, result['weights'], con_names, output_file)
    print(f"\nOutput saved to: {output_file}")

    n_correct, n_total, acc = compute_accuracy(preds)
    mae, _per_tableau = compute_errors(preds)
    print(f"Accuracy (|error| <= 0.05): {n_correct}/{n_total} = {acc:.2f}%")
    print(f"Mean absolute error: {mae:.5f}")

    return result, preds


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    run_model(path)
