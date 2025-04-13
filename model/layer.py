import numpy as np

class Layer:
    """
    A single feedforward layer containing:
      - weights
      - biases
      - velocity terms for momentum updates
    """
    def __init__(self, in_size, out_size):
        # Initialize weights and biases
        self.weights = np.random.uniform(
                low=-np.sqrt(6. / (in_size + out_size)),
                high=np.sqrt(6. / (in_size + out_size)),
                size=(in_size, out_size)
        )
        self.biases  = np.zeros(out_size)

        # Velocity terms for momentum
        self.v_weights = np.zeros_like(self.weights)
        self.v_biases  = np.zeros_like(self.biases)

    def forward(self, x):
        """
        Computes: z = xW + b
        """
        return np.dot(x, self.weights) + self.biases

    def update(self, dw, db, learning_rate, momentum):
        """
        Updates the layer’s parameters (weights, biases) with momentum.
        """
        self.v_weights = momentum * self.v_weights + dw
        self.weights   -= learning_rate * self.v_weights

        self.v_biases  = momentum * self.v_biases + db
        self.biases    -= learning_rate * self.v_biases

    def decay_weight(self, decay_complement):
        """
        Applies weight decay to the weights (not the biases).
        """
        self.weights *= decay_complement
