import numpy as np

### ================================================================================
### ACTIVATION FUNCTIONS

def relu(s): 
    return np.maximum(s, 0)

def relu_prime(s):
    return np.where(s > 0, 1.0, 0.0)

def leaky_relu(s, alpha=0.01):
    return np.where(s >= 0, s, alpha * s)

def leaky_relu_prime(s, alpha=0.01):
    return np.where(s > 0, 1.0, alpha)

def sigmoid(s):
    return 1 / (1 + np.exp(-s))

def sigmoid_prime(s):
    return np.exp(-s) / ((1+np.exp(-s)) * (1 + np.exp(-s)))

def tanh(s):
    return (np.exp(s) - np.exp(-s)) / (np.exp(s) + np.exp(-s))

def tanh_prime(s):
    return 1 - ( np.tanh(s) ** 2 )

def test_val(s):
    print(f"RELU: {relu(s)} Prime: {relu_prime(s)}")
    print(f"Leaky RELU: {leaky_relu(s)} Prime: {leaky_relu_prime(s)}")
    print(f"Sigmoid: {sigmoid(s)} Prime: {sigmoid_prime(s)}")
    print(f"Tanh: {tanh(s)} Prime: {tanh_prime(s)}")

activation_functions = {
    "relu": relu,
    "leaky_relu": leaky_relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
}

activation_primes = {
    "relu": relu_prime,
    "leaky_relu": leaky_relu_prime,
    "sigmoid": sigmoid_prime,
    "tanh": tanh_prime,
}

### ================================================================================
# TRAINING ERROR FUNCTIONS
def euclidian(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred)**2))

def cross_entropy(y_true, y_pred, epsilon=1e-12):
    # clip predictions to [epsilon, 1]
    y_pred_clipped = np.clip(y_pred, epsilon, 1.0)
    return -np.sum(y_true * np.log(y_pred_clipped))

def cross_entropy_symmetric(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred) + y_pred* np.log(y_true))

training_error = {
    "euclidian": euclidian,
    "cross_entropy": cross_entropy,
    "cross_entropy_symmetric": cross_entropy_symmetric
}

### ================================================================================
# SOFTMAX Functions
def softmax(vector):
    return np.exp(vector)/np.sum(np.exp(vector))

def softmax_shifted(vector): # Subtract max to improve numerical stability
    shift = np.max(vector)
    shifted_vector = vector - shift
    exp_vec = np.exp(shifted_vector)
    return exp_vec / np.sum(exp_vec)

output_transformation = {
    "softmax": softmax,
    "softmax_shifted": softmax_shifted
}
