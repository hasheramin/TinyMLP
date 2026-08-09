import math
from tinymlp.core.value import Value


def binary_cross_entropy(y_pred, y_true, eps: float = 1e-7):
    """
    Computes Binary Cross Entropy (BCE) loss for probability inputs in [0, 1]:
    loss = -1/N * sum(y * log(p) + (1 - y) * log(1 - p))

    Handles boundary probabilities (0.0 and 1.0) numerically while preserving
    autograd computational graph and gradient propagation.
    """
    if not isinstance(y_pred, list):
        y_pred = [y_pred]
    if not isinstance(y_true, list):
        y_true = [y_true]

    if len(y_pred) != len(y_true):
        raise ValueError(
            f"y_pred and y_true must have the same length, got {len(y_pred)} and {len(y_true)}"
        )

    total_loss = Value(0.0)
    for pred, target in zip(y_pred, y_true):
        pred_node = pred if isinstance(pred, Value) else Value(pred)
        target_node = target if isinstance(target, Value) else Value(target)

        p_data = pred_node.data
        t_data = target_node.data

        if p_data < 0.0 or p_data > 1.0:
            raise ValueError(
                f"Invalid prediction probability {p_data}: predictions must be in range [0, 1]"
            )
        if t_data < 0.0 or t_data > 1.0:
            raise ValueError(
                f"Invalid target probability {t_data}: targets must be in range [0, 1]"
            )

        # Clamped probability values for safe log evaluation
        p_clamped = max(eps, min(1.0 - eps, p_data))
        one_minus_p_clamped = max(eps, min(1.0 - eps, 1.0 - p_data))

        # Direct autograd log nodes with clipping protection
        p_log = Value(math.log(p_clamped), (pred_node,), "log")

        def _backward_p_log(node=pred_node, p_c=p_clamped, p_l=p_log):
            node.grad += (1.0 / p_c) * p_l.grad

        p_log._backward = _backward_p_log

        one_minus_p_log = Value(math.log(one_minus_p_clamped), (pred_node,), "log")

        def _backward_one_minus_p_log(node=pred_node, omp_c=one_minus_p_clamped, omp_l=one_minus_p_log):
            node.grad -= (1.0 / omp_c) * omp_l.grad

        one_minus_p_log._backward = _backward_one_minus_p_log

        term1 = target_node * p_log
        term2 = (Value(1.0) - target_node) * one_minus_p_log
        sample_loss = -(term1 + term2)
        total_loss = total_loss + sample_loss

    return total_loss / len(y_pred)
