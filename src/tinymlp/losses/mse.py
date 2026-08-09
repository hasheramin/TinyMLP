from tinymlp.core.value import Value


def mse_loss(y_pred, y_true):
    """
    Computes Mean Squared Error (MSE) loss.
    Accepts single values or lists of values.
    """
    if not isinstance(y_pred, list):
        y_pred = [y_pred]
    if not isinstance(y_true, list):
        y_true = [y_true]

    assert len(y_pred) == len(y_true), "y_pred and y_true must have the same length"

    total_loss = Value(0.0)
    for pred, target in zip(y_pred, y_true):
        target_val = target if isinstance(target, Value) else Value(target)
        diff = pred - target_val
        total_loss = total_loss + (diff ** 2)

    return total_loss / len(y_pred)
