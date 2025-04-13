import numpy as np

class Layer:
    def __init__(
        self,
        in_size,
        out_size,
        activation=None,
        activation_prime=None,
        output_transform=None,
        momentum=0.9,
        dropout_rate=0.0,
        is_output=False
    ):
        """
        A single fully-connected layer:
          W shape = (in_size, out_size)
          b shape = (out_size,)

        If is_output=True, we apply 'output_transform' instead of 'activation'
        and typically skip dropout for the output layer.
        """
        # Initialize weights and biases
        self.W = np.random.uniform(low=-1.0, high=1.0, size=(in_size, out_size))
        self.b = np.zeros(out_size)

        # Initialize velocity terms for momentum
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

        # Store functions
        self.activation = activation           # e.g. relu
        self.activation_prime = activation_prime
        self.output_transform = output_transform
        self.momentum = momentum
        self.dropout_rate = dropout_rate
        self.is_output = is_output

        # Internal buffers for forward/backward
        self.z = None      # pre-activation
        self.a = None      # post-activation
        self.mask = None   # dropout mask (for hidden layers)

    def forward(self, x, training=True):
        """
        Forward pass:
          z = x @ W + b
          a = activation(z) or output_transform(z) if is_output
          If dropout and not is_output: apply dropout to a
        """
        self.z = np.dot(x, self.W) + self.b  # shape: (batch, out_size) or just (out_size,) for single sample

        if self.is_output and self.output_transform:
            # e.g. softmax
            self.a = self.output_transform(self.z)
        else:
            # hidden layer activation
            self.a = self.activation(self.z)

            # apply dropout if training
            if training and self.dropout_rate > 0.0:
                keep_prob = 1.0 - self.dropout_rate
                self.mask = (np.random.rand(*self.a.shape) < keep_prob).astype(np.float32)
                self.a = (self.a * self.mask) / keep_prob
            else:
                self.mask = None

        return self.a

    def backward(self, delta_in, x_in, learning_rate):
        """
        Backward pass for one layer.
        - delta_in: gradient wrt *this layer's output* (size out_size)
        - x_in: the input that was fed *into* this layer (size in_size)
        - Update self.W, self.b with momentum
        - Return delta_out, which is the gradient wrt the *previous* layer’s output
        """
        mu = self.momentum

        # 1) If not output layer, multiply by activation_prime(z)
        #    If it was output layer with cross-entropy+softmax, you might do delta_in as is (a_output - y).
        #    But let's assume the "chain rule" upstream already accounted for that.
        if not self.is_output:
            # derivative wrt z
            delta_z = delta_in * self.activation_prime(self.z)

            # also re-apply the dropout mask if used
            if self.mask is not None:
                keep_prob = 1.0 - self.dropout_rate
                delta_z = (delta_z * self.mask) / keep_prob
        else:
            # output layer: we assume delta_in is already dL/dz
            delta_z = delta_in

        # 2) Compute gradients wrt W, b
        # For a single sample, x_in is shape (in_size,), delta_z is (out_size,).
        dW = np.outer(x_in, delta_z)  # (in_size, out_size)
        db = delta_z                  # (out_size,)

        # 3) Momentum update
        self.vW = mu * self.vW + dW
        self.vb = mu * self.vb + db

        # 4) Gradient descent step
        self.W -= learning_rate * self.vW
        self.b -= learning_rate * self.vb

        # 5) Return delta for the previous layer: dL/dx
        #    shape: (in_size,) = W @ delta_z
        delta_out = np.dot(self.W, delta_z)   # passing gradient upstream
        return delta_out

    def apply_weight_decay(self, weight_decay):
        """
        Multiply W by (1 - weight_decay), a simplified L2 approach.
        Typically we do *not* decay biases.
        """
        if weight_decay > 0.0:
            decay_complement = 1.0 - weight_decay
            self.W *= decay_complement
