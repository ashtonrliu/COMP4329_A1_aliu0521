import numpy as np

class BatchNorm(object):
    """
    Batch Normalization for 1D inputs (i.e., a single activation vector)
    with gradient clipping in the update step.

    Attributes:
        gamma (np.array):    Learnable scale parameter for BN (initialized to 1).
        beta (np.array):     Learnable shift parameter for BN (initialized to 0).
        running_mean (np.array): Running average of means (for inference).
        running_var (np.array):  Running average of variances (for inference).
        bn_momentum (float): Momentum term for updating running stats.
        eps (float):         Small constant added to variance for numerical stability.
        dgamma (np.array):   Accumulated gradient wrt gamma (reset after each update).
        dbeta (np.array):    Accumulated gradient wrt beta (reset after each update).
        x_hat (np.array):    Saved normalized input for backprop.
        inv_std (float):     Saved inverse std for backprop.

        clip_value (float):  Threshold for gradient clipping in L2 norm.
    """
    def __init__(self, dim, bn_momentum=0.9, eps=1e-5, clip_value=1.0):
        """
        Initializes Batch Normalization parameters.

        Parameters:
        - dim (int): number of output units to normalize.
        - bn_momentum (float): momentum for updating running mean/var.
        - eps (float): small constant to avoid division by zero in variance.
        - clip_value (float): L2-norm threshold for gradient clipping.
        """

        self.dim = dim
        self.bn_momentum = bn_momentum
        self.eps = eps
        self.clip_value = clip_value  # For gradient clipping

        # Learnable scale and shift
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta  = np.zeros(dim, dtype=np.float32)

        # Running (moving) averages for inference
        self.running_mean = np.zeros(dim, dtype=np.float32)
        self.running_var  = np.ones(dim,  dtype=np.float32)

        # Grad accumulators
        self.dgamma = np.zeros(dim, dtype=np.float32)
        self.dbeta  = np.zeros(dim, dtype=np.float32)

        # Saved for backward
        self.x_hat   = None  # normalized input
        self.inv_std = None  # 1 / sqrt(var + eps)

    def forward(self, x, training=True):
        """
        Forward pass for Batch Normalization. For single-sample usage,
        we approximate mean ~ x, var ~ 0, then use 'running_mean', 'running_var'
        for a partial fix of internal covariate shift.
        """
        if training:
            mean = x
            var  = np.zeros_like(x)

            self.running_mean = (
                self.bn_momentum * self.running_mean
                + (1.0 - self.bn_momentum) * mean
            )

            est_var = var + 1e-5
            self.running_var = (
                self.bn_momentum * self.running_var
                + (1.0 - self.bn_momentum) * est_var
            )
            self.running_var = np.maximum(self.running_var, 1e-5)

            x_centered = x - self.running_mean
            inv_std = 1.0 / np.sqrt(self.running_var + self.eps)
        else:
            x_centered = x - self.running_mean
            inv_std = 1.0 / np.sqrt(self.running_var + self.eps)

        x_hat = x_centered * inv_std

        # Save for backprop
        self.x_hat = x_hat
        self.inv_std = inv_std

        # BN output
        out = self.gamma * x_hat + self.beta
        # print(self.running_var)
        return out

    def backward(self, grad_output):
        """
        Backprop for BN in single-sample (or small-batch) mode:
            dgamma += sum(grad_output * x_hat)
            dbeta  += sum(grad_output)
            dx = grad_output * gamma * inv_std
        """
        self.dgamma += grad_output * self.x_hat
        self.dbeta  += grad_output

        dx = grad_output * self.gamma * self.inv_std
        return dx

    def update(self, lr):
        """
        Update gamma and beta with gradient clipping (L2 norm).
        Resets dgamma and dbeta after the update.

        Parameters:
        - lr (float): learning rate for BN parameters.
        """
        # Combine the BN param gradients for a single L2 norm
        grad_vec = np.concatenate([self.dgamma.ravel(), self.dbeta.ravel()])
        grad_norm = np.linalg.norm(grad_vec, ord=2)

        # Clip if exceeding self.clip_value
        if grad_norm > self.clip_value:
            scale = self.clip_value / grad_norm
            self.dgamma *= scale
            self.dbeta  *= scale

        # Gradient descent on gamma, beta
        self.gamma -= lr * self.dgamma
        self.beta  -= lr * self.dbeta

        # Reset grads
        self.dgamma.fill(0.0)
        self.dbeta.fill(0.0)
