import numpy as np

def batch_normalization(
    X: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray = None,
    running_var: np.ndarray = None,
    momentum: float = 0.1,
    epsilon: float = 1e-5,
    training: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Batch Normalization on BCHW input.
    
    Args:
        X: Input array of shape (B, C, H, W)
        gamma: Scale parameter of shape (1, C, 1, 1)
        beta: Shift parameter of shape (1, C, 1, 1)
        running_mean: Running mean for inference, shape (1, C, 1, 1)
        running_var: Running variance for inference, shape (1, C, 1, 1)
        momentum: Momentum for updating running statistics (following PyTorch convention)
        epsilon: Small constant for numerical stability
        training: If True, use batch statistics; if False, use running statistics
    
    Returns:
        Tuple of (normalized_output, updated_running_mean, updated_running_var)
    """
    # Your code here

    # Initialize running statistics
    C = X.shape[1]

    if running_mean is None:
        running_mean = np.zeros((1, C, 1, 1))

    if running_var is None:
        running_var = np.ones((1, C, 1, 1))

    if training:
        # Compute mean and variance for each channel across B, H, W
        batch_mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
        batch_var = np.var(X, axis=(0, 2, 3), keepdims=True)

        # Normalize using current batch statistics
        X = (X - batch_mean) / np.sqrt(batch_var + epsilon)

        # Update running statistics
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        running_var = (1 - momentum) * running_var + momentum * batch_var

    else:
        # Normalize using stored running statistics
        X = (X - running_mean) / np.sqrt(running_var + epsilon)

    # Apply learnable scale and shift
    X = X * gamma + beta

    return X, running_mean, running_var
