import numpy as np


def adam_update(
    theta: np.ndarray,
    grad: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Update moments
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad**2)
    # Compute bias corrected estimates
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    # Update param
    theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta, m, v


def adamw_update(
    theta: np.ndarray,
    grad: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    weight_decay: float = 0.0,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # AdamW update: same moment updates using grad of loss only (no weight decay term added)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad**2)
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    # Gradient-based parameter update
    theta_prev = theta.copy()
    theta = theta - lr * m_hat / (np.sqrt(v_hat) + eps)
    # decoupled weight decay step (do not apply to bias terms)
    if weight_decay != 0:
        theta = theta - lr * weight_decay * theta_prev
    return theta, m, v


if __name__ == "__main__":
    # Adam usage example (one step):
    theta = np.array([0.0, 0.0])
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    t = 1
    grad = np.array([0.1, -0.2])  # random exampel gradient
    theta, m, v = adam_update(theta, grad, m, v, t, lr=0.001)
    # Can you guess the output??
    print("Adam update (one step):")
    # Can you guess the output??"
    print(f"theta = {theta}, m = {m}, v = {v}\n")

    # AdamW usage example (one step):
    theta = np.array([0.0, 0.0])
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    grad = np.array([0.1, -0.2])
    theta, m, v = adamw_update(theta, grad, m, v, t=1, lr=0.001, weight_decay=0.01)
    print("AdamW update (one step):")
    # Can you guess the output??"
    print(f"theta = {theta}, m = {m}, v = {v}")
