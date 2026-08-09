import sys
import random
import math
from pathlib import Path

# Add src to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tinymlp.nn.mlp import MLP
from tinymlp.losses.mse import mse_loss
from tinymlp.optim.sgd import SGD


def main():
    random.seed(42)
    print("=== TinyMLP: 1D Nonlinear Function Regression ===")
    print("Target function: y = sin(pi * x)")

    # Generate synthetic training points x in [-1, 1]
    X_train = [[x / 10.0] for x in range(-10, 11, 2)]
    y_train = [math.sin(math.pi * x[0]) for x in X_train]

    # Model: 1 input -> [8, 8] hidden (tanh) -> 1 output (linear)
    model = MLP(1, [8, 8, 1], activations=['tanh', 'tanh', 'linear'])
    optimizer = SGD(model.parameters(), lr=0.05)

    print("Training model...")
    epochs = 300

    for epoch in range(1, epochs + 1):
        preds = [model(x) for x in X_train]
        loss = mse_loss(preds, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 60 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | MSE Loss: {loss.data:.6f}")

    print("\n=== Predictions vs Ground Truth ===")
    print(f"{'x':>8} | {'Predicted':>10} | {'True y':>10} | {'Error':>10}")
    print("-" * 48)
    for x_i, target in zip(X_train, y_train):
        pred_val = model(x_i).data
        err = abs(pred_val - target)
        print(f"{x_i[0]:8.2f} | {pred_val:10.4f} | {target:10.4f} | {err:10.4f}")


if __name__ == "__main__":
    main()
