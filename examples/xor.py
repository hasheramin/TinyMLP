import sys
import random
from pathlib import Path

# Add src to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tinymlp.nn.mlp import MLP
from tinymlp.losses.mse import mse_loss
from tinymlp.optim.sgd import SGD


def main():
    random.seed(42)
    print("=== TinyMLP: XOR Problem Solver ===")

    # Dataset: XOR truth table
    X = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
    y = [0.0, 1.0, 1.0, 0.0]

    # Model: 2 inputs -> 4 hidden neurons (tanh) -> 1 output neuron (sigmoid)
    model = MLP(2, [4, 1], activations=['tanh', 'sigmoid'])
    optimizer = SGD(model.parameters(), lr=0.5)

    print("Training model...")
    epochs = 200

    for epoch in range(1, epochs + 1):
        # Forward pass for all samples
        preds = [model(x) for x in X]
        loss = mse_loss(preds, y)

        # Backward pass & optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 40 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {loss.data:.6f}")

    print("\n=== Final Predictions ===")
    for x_i, target in zip(X, y):
        pred_val = model(x_i).data
        rounded = round(pred_val)
        print(f"Input: {x_i} -> Predicted: {pred_val:.4f} (rounded: {rounded}) | Target: {target}")


if __name__ == "__main__":
    main()
