import numpy as np

from utils.functions import (
    activation_functions, 
    activation_primes, 
    training_error, 
    output_transformation
)

from model.layer import Layer 

class Model:
    def __init__(
        self, 
        activation="relu", 
        error="cross_entropy", 
        output_transform="softmax_shifted", 
        *, 
        i_size=128, 
        h1_size=92, 
        h2_size=68, 
        o_size=10, 
        learning_rate=0.01, 
        weight_decay=0.02, 
        momentum=0.9, 
        dropout_rate=0.5
    ):
        """
        Initializes the MLP model with two hidden layers and one output layer,
        using the given hyperparameters. 
        """
        # Cache the activation, error, transform, etc.
        self.activation_name = activation
        self.activation = activation_functions[activation]
        self.activation_prime = activation_primes[activation]
        self.error = training_error[error]
        self.output_transform = output_transformation[output_transform]

        # Create three layers
        self.hidden_1 = Layer(i_size, h1_size)
        self.hidden_2 = Layer(h1_size, h2_size)
        self.output   = Layer(h2_size, o_size)

        self.learning_rate = learning_rate 
        self.weight_decay  = weight_decay  # Set to 0 to turn off
        self.momentum      = momentum      # Set to 0 to turn off
        self.dropout_rate  = dropout_rate

        # Pre-cache keep_prob for dropout (1 - dropout_rate)
        self._keep_prob = 1.0 - dropout_rate

    def forward(self, x, training=True):
        """
        Given an input vector x, performs a forward pass through the neural network.
        Returns the outputs (z) and activations (a) for each layer.
        """
        # First hidden layer
        z_hidden_1 = self.hidden_1.forward(x)
        a_hidden_1 = self.activation(z_hidden_1)
        a_hidden_1, mask1 = self.forward_dropout(a_hidden_1, training=training)

        # Second hidden layer
        z_hidden_2 = self.hidden_2.forward(a_hidden_1)
        a_hidden_2 = self.activation(z_hidden_2)
        a_hidden_2, mask2 = self.forward_dropout(a_hidden_2, training=training)

        # Output layer
        z_output = self.output.forward(a_hidden_2)
        a_output = self.output_transform(z_output)

        # Save dropout masks so backward can use them
        self.mask1 = mask1
        self.mask2 = mask2

        return z_hidden_1, a_hidden_1, z_hidden_2, a_hidden_2, z_output, a_output

    def forward_dropout(self, a_input, training=True):
        """
        Applies dropout to 'a_input'.
        dropout_rate=0.5 => we drop 50% of the neurons (elements).
        If not training, returns a_input unchanged.
        """
        if not training:
            return a_input, None

        keep_prob = self._keep_prob
        shape = a_input.shape
        mask = (np.random.rand(*shape) < keep_prob).astype(np.float32)
        a_dropped = (a_input * mask) / keep_prob
        return a_dropped, mask

    def backward_dropout(self, delta, mask, keep_prob):
        """
        Zero out the gradient for dropped neurons during backprop.
        """
        if mask is None:
            return delta
        return (delta * mask) / keep_prob

    def backward(self, x, y, z_hidden_1, a_hidden_1, z_hidden_2, a_hidden_2, z_output, a_output):
        """
        Performs backpropagation and updates the network weights & biases. 
        No BN references. 
        """
        hidden_1 = self.hidden_1
        hidden_2 = self.hidden_2
        output   = self.output

        activation_prime = self.activation_prime
        backward_dropout = self.backward_dropout

        w_out = output.weights
        w_h2  = hidden_2.weights
        w_h1  = hidden_1.weights
        keep_prob = self._keep_prob

        # 1) Output layer delta
        delta_output = a_output - y  # shape: (o_size,)

        dw_output = np.outer(a_hidden_2, delta_output) 
        db_output = delta_output

        # 2) Backprop to second hidden layer
        d_a_hidden_2 = np.dot(w_out, delta_output)  # shape: (h2_size,)
        delta_hidden_2 = d_a_hidden_2 * activation_prime(z_hidden_2)
        delta_hidden_2 = backward_dropout(delta_hidden_2, self.mask2, keep_prob)

        dw_hidden_2 = np.outer(a_hidden_1, delta_hidden_2)
        db_hidden_2 = delta_hidden_2

        # 3) Backprop to first hidden layer
        d_a_hidden_1 = np.dot(w_h2, delta_hidden_2) 
        delta_hidden_1 = d_a_hidden_1 * activation_prime(z_hidden_1)
        delta_hidden_1 = backward_dropout(delta_hidden_1, self.mask1, keep_prob)

        dw_hidden_1 = np.outer(x, delta_hidden_1)
        db_hidden_1 = delta_hidden_1

        # 4) Update each layer using momentum
        lr = self.learning_rate
        mu = self.momentum

        # Output
        output.update(dw_output, db_output, lr, mu)
        # Hidden_2
        hidden_2.update(dw_hidden_2, db_hidden_2, lr, mu)
        # Hidden_1
        hidden_1.update(dw_hidden_1, db_hidden_1, lr, mu)

    def decay_weight(self):
        """
        Weight decay (L2-style).
        """
        decay_complement = 1.0 - self.weight_decay
        self.hidden_1.decay_weight(decay_complement)
        self.hidden_2.decay_weight(decay_complement)
        self.output.decay_weight(decay_complement)

    def train(self, dataset, labels, X_test, y_test, epoch=50):
        """
        Single-sample training loop. 
        For each epoch, we train sample-by-sample, then print accuracy.
        """
        dataset_len = len(dataset)
        if dataset_len != len(labels):
            print("ERROR: dataset and labels mismatch")
            return

        return_data = []
        for e in range(epoch):
            total_loss = 0.0
            for i in range(dataset_len):
                x = dataset[i]
                y = labels[i]

                # Forward pass
                z_h1, a_h1, z_h2, a_h2, z_out, a_out = self.forward(x, training=True)

                # Loss
                loss_val = self.error(y, a_out)
                total_loss += loss_val

                # Backprop
                self.backward(x, y, z_h1, a_h1, z_h2, a_h2, z_out, a_out)

            # Weight decay
            self.decay_weight()

            # Test after epoch
            accuracy = self.test_accuracy(X_test, y_test)
            avg_loss = total_loss / dataset_len
            print(f"Epoch {e+1}/{epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            return_data.append((e, avg_loss, accuracy))
        
        return return_data

    def train_batch(self, X, y, X_test, y_test, epochs=20, batch_size=32):
        """
        Trains the model on the data (X, y) using mini-batch gradient descent.
        Logs accuracy similarly to the single-sample 'train' method.
        """
        num_samples = len(X)
        return_data = []

        for e in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            total_loss = 0.0

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_X = X_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]

                batch_len = len(batch_X)
                # We'll do single-sample forward/back for each item in batch
                batch_loss_sum = 0.0

                for i in range(batch_len):
                    x_i = batch_X[i]
                    y_i = batch_y[i]

                    # Forward
                    z_h1, a_h1, z_h2, a_h2, z_out, a_out = self.forward(x_i, training=True)
                    # Loss
                    loss_val = self.error(y_i, a_out)
                    batch_loss_sum += loss_val

                    # Backprop
                    self.backward(x_i, y_i, z_h1, a_h1, z_h2, a_h2, z_out, a_out)

                total_loss += batch_loss_sum

            # Weight decay
            self.decay_weight()

            # Evaluate after epoch
            accuracy = self.test_accuracy(X_test, y_test)
            avg_loss = total_loss / num_samples
            print(f"Epoch {e+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            return_data.append((e, avg_loss, accuracy))

        return return_data

    def run_train(self, dataset, labels, X_test, y_test, epoch=50, batch_size=None):
        """
        Runs training. If batch_size is None or <=1, we do single-sample training.
        Otherwise, we do mini-batch training.
        """
        if batch_size is None or batch_size <= 1:
            # Use single-sample training
            return self.train(dataset, labels, X_test, y_test, epoch=epoch)
        else:
            # Use mini-batch training
            print("batch training")
            return self.train_batch(dataset, labels, X_test, y_test, epochs=epoch, batch_size=batch_size)

    def test_model(self, X_test, y_test):
        """
        Evaluates the model on the provided test set, returning:
        - Accuracy
        - Macro-averaged Precision
        - Macro-averaged Recall
        - Macro-averaged F1-score
        - Average Cross-Entropy Loss
        Expects y_test to be one-hot encoded.
        """
        dataset_length = len(X_test)
        if dataset_length == 0:
            print("Warning: test set is empty.")
            return None

        # 1. Gather predictions and compute cross-entropy loss
        y_true_list = []
        y_pred_list = []
        total_loss = 0.0

        for i in range(dataset_length):
            x = X_test[i]
            # Convert one-hot test label to integer
            true_label = np.argmax(y_test[i])
            # Forward pass with dropout disabled (training=False)
            _, _, _, _, _, a_out = self.forward(x, training=False)
            pred_label = np.argmax(a_out)

            y_true_list.append(true_label)
            y_pred_list.append(pred_label)

            # Accumulate cross-entropy loss
            loss_val = self.error(y_test[i], a_out)
            total_loss += loss_val

        avg_cross_entropy = total_loss / dataset_length

        # 2. Construct confusion matrix
        num_classes = y_test.shape[1]
        confusion_mat = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true_list, y_pred_list):
            confusion_mat[t, p] += 1

        # 3. Compute accuracy from confusion matrix
        correct_predictions = np.trace(confusion_mat)
        accuracy = correct_predictions / dataset_length

        # 4. Compute macro-averaged precision, recall, and F1
        precisions = []
        recalls = []
        f1_scores = []

        for c in range(num_classes):
            tp = confusion_mat[c, c]
            fp = np.sum(confusion_mat[:, c]) - tp
            fn = np.sum(confusion_mat[c, :]) - tp

            # Precision
            precision_c = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            # Recall
            recall_c = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # F1
            if (precision_c + recall_c) > 0:
                f1_c = 2 * precision_c * recall_c / (precision_c + recall_c)
            else:
                f1_c = 0.0

            precisions.append(precision_c)
            recalls.append(recall_c)
            f1_scores.append(f1_c)

        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        macro_f1 = np.mean(f1_scores)

        # 5. Return or print the results
        results = {
            "accuracy": accuracy,
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1,
            "cross_entropy": avg_cross_entropy
        }

        return results
    
    def test_accuracy(self, X_test, y_test):
        """
        Evaluates the model on the test set and returns only the overall accuracy.
        Expects y_test to be one-hot encoded.
        """
        dataset_length = len(X_test)
        if dataset_length == 0:
            print("Warning: test set is empty.")
            return None

        correct = 0
        for i in range(dataset_length):
            x = X_test[i]
            # Convert one-hot test label to integer
            true_label = np.argmax(y_test[i])

            # Forward pass with dropout disabled (training=False)
            _, _, _, _, _, a_out = self.forward(x, training=False)
            pred_label = np.argmax(a_out)

            # Count correct predictions
            if pred_label == true_label:
                correct += 1

        # Calculate accuracy
        accuracy = correct / dataset_length
        return accuracy


    def predict(self, x, training=True):
        """
        Single-sample predict. Returns the argmax class from the final layer's output.
        """
        _, _, _, _, _, a_output = self.forward(x, training=training)
        max_class = np.argmax(a_output)
        return max_class

    def get_parameters(self):
        return { 
            "activation_function": self.activation.__name__,
            "error": self.error.__name__,
            "output_transformation": self.output_transform.__name__
        }

    def __repr__(self):
        return (
            f"activation_function:\t{self.activation.__name__}\n"
            f"error:\t\t\t{self.error.__name__}\n"
            f"output_transformation:\t{self.output_transform.__name__}"
        )
