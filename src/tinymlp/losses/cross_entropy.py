from tinymlp.core.value import Value


def binary_cross_entropy(y_pred, y_true, eps: float = 1e-7):
    """
    Computes Binary Cross Entropy (BCE) loss:
    loss = -1/N * sum(y * log(p + eps) + (1 - y) * log(1 - p + eps))
    """
    if not isinstance(y_pred, list):
        y_pred = [y_pred]
    if not isinstance(y_true, list):
        y_true = [y_true]

    assert len(y_pred) == len(y_true), "y_pred and y_true must have the same length"

    total_loss = Value(0.0)
    for pred, target in zip(y_pred, y_true):
        # Clip or bound predictions for numerical stability
        pred_val = pred
        target_val = target if isinstance(target, Value) else Value(target)
        
        # - (y * log(p) + (1 - y) * log(1 - p))
        p = pred_val + eps
        one_minus_p = (Value(1.0) - pred_val) + eps

        term1 = target_val * p.log()
        term2 = (Value(1.0) - target_val) * one_minus_p.log()
        sample_loss = -(term1 + term2)
        total_loss = total_loss + sample_loss

    return total_loss / len(y_pred)
