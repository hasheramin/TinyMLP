import sys
from pathlib import Path

# Add src to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tinymlp.core.value import Value
from tinymlp.nn.mlp import MLP
from tinymlp.losses.mse import mse_loss


def main():
    print("=== TinyMLP: Basic Forward & Backward Pass Demo ===")
    
    # Define a small 3-layer MLP: 3 inputs -> [4, 4] hidden -> 1 output
    model = MLP(3, [4, 4, 1], activations=['relu', 'relu', 'linear'])
    print("Model Architecture:")
    print(model)
    print(f"Total Parameters: {len(model.parameters())}\n")

    # Sample input
    x = [Value(2.0, label='x0'), Value(-1.0, label='x1'), Value(0.5, label='x2')]
    target = 1.0

    print(f"Input: {[v.data for v in x]}")
    print(f"Target: {target}")

    # Forward pass
    output = model(x)
    print(f"Initial Model Output: {output.data:.4f}")

    # Compute loss
    loss = mse_loss(output, target)
    print(f"Initial Loss: {loss.data:.4f}")

    # Backward pass
    model.zero_grad()
    loss.backward()

    print("\nGradient updates computed for parameters (first 5 shown):")
    for i, p in enumerate(model.parameters()[:5]):
        print(f"  Param {i}: data={p.data:.4f}, grad={p.grad:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
