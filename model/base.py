from utils.functions import activation_functions, activation_primes, training_error, output_transformation
import random
import numpy as np


class Model:
    def __init__(self, activation="relu", error="cross_entropy", output_transform="softmax_shifted", i_size=128, h1_size=92, h2_size=68, o_size=10, learning_rate=0.01, weight_decay=0.02, momentum=0.9):
        # Default Heuristic for hidden layer size
        # h1_size = (i_size + o_size) * 2/3 = 138 * 2/3 = 92
        # h2_size = (h1_size + o_size ) * 2/3 = 68

        self.activation_name = activation
        self.activation = activation_functions[activation]
        self.activation_prime = activation_primes[activation]
        self.error = training_error[error]
        self.output_transform = output_transformation[output_transform]

        # MLP Architecture: 2 Hidden Layers
        self.w_hidden_1 = np.random.uniform(low=-1, high=1, size=(i_size, h1_size)) 
        self.b_hidden_1 = np.zeros(h1_size)

        self.w_hidden_2 = np.random.uniform(low=-1, high=1, size=(h1_size, h2_size)) 
        self.b_hidden_2 = np.zeros(h2_size)
        
        self.w_output = np.random.uniform(low=-1, high=1, size=(h2_size, o_size)) 
        self.b_output = np.zeros(o_size)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum

    def forward(self, x):
        """
        Given an input vector x, performs a forward pass through the neural network, returns the outputs (z) and activations (a) for each layer
        """
        z_hidden_1 = np.dot(x, self.w_hidden_1) + self.b_hidden_1
        a_hidden_1 = self.activation(z_hidden_1)

        z_hidden_2 = np.dot(a_hidden_1, self.w_hidden_2) + self.b_hidden_2
        a_hidden_2 = self.activation(z_hidden_2)

        z_output = np.dot(a_hidden_2, self.w_output) + self.b_output
        a_output = self.output_transform(z_output)

        return z_hidden_1, a_hidden_1, z_hidden_2, a_hidden_2, z_output, a_output

    def backward(self, x, y, z_hidden_1, a_hidden_1, z_hidden_2, a_hidden_2, z_output, a_output):
        """
        Perorms a backwards propagation through the neural network, updating the weights for a given loss
        """

        # Delta
        delta_output = a_output - y

        # Gradients of output layer
        dw_output = np.outer(a_hidden_2, delta_output)  # (h2_size, o_size)
        db_output = delta_output                        # (o_size,)

        # Backprop to second hidden layer
        d_a_hidden_2 = np.dot(self.w_output, delta_output)  # shape: (h2_size,)
        delta_hidden_2 = d_a_hidden_2 * self.activation_prime(z_hidden_2)

        dw_hidden_2 = np.outer(a_hidden_1, delta_hidden_2)  # (h1_size, h2_size)
        db_hidden_2 = delta_hidden_2                        # (h2_size,)

        # Backprop to first hidden layer
        d_a_hidden_1 = np.dot(self.w_hidden_2, delta_hidden_2)  # shape: (h1_size,)
        delta_hidden_1 = d_a_hidden_1 * self.activation_prime(z_hidden_1)

        dw_hidden_1 = np.outer(x, delta_hidden_1)  # (i_size, h1_size)
        db_hidden_1 = delta_hidden_1              # (h1_size,)

        # Update weights
        lr = self.learning_rate
        self.w_output  -= lr * dw_output
        self.b_output  -= lr * db_output
        self.w_hidden_2 -= lr * dw_hidden_2
        self.b_hidden_2 -= lr * db_hidden_2
        self.w_hidden_1 -= lr * dw_hidden_1
        self.b_hidden_1 -= lr * db_hidden_1

        return

    def decay_weight(self):
        """
        Applies weight decay by multiplying by the weight_decay constant on all the weights.

        Using decay complement as an implementation of the Weight Decay Engineers' view which is a simplified version of L2 regularisation.
        """

        decay_complement = 1.0 - self.weight_decay
        self.w_hidden_1 *= decay_complement
        self.w_hidden_2 *= decay_complement
        self.w_output *= decay_complement

    def train(self, dataset, labels, epoch=50):
        """
        Initialises the epoch and batch training for the system

        Where s in the input for the system and y is the label
        """
        dataset_len = len(dataset)

        if dataset_len != len(labels):
            print("ERROR: dataset and labels are not the same size")
            return
        
        for e in range(epoch):
            total_loss = 0.0

            for i in range(dataset_len):
                x = dataset[i]
                y = labels[i]
                # print(x, y)
        
                z_hidden_1, a_hidden_1, z_hidden_2, a_hidden_2, z_output, a_output = self.forward(x)

                # Compute loss
                loss_val = self.error(y, a_output)
                # print(y, a_output)

                total_loss += loss_val

                self.backward(x, y, z_hidden_1, a_hidden_1, z_hidden_2, a_hidden_2, z_output, a_output)

            self.decay_weight()

            print(f"Epoch {e+1}/{epoch}, Loss: {total_loss / dataset_len:.4f}")

    def get_parameters(self):
        return { 
            "activation_function": self.activation.__name__,
            "error": self.error.__name__,
            "output_transformation": self.output_transform.__name__
        }
    
    def predict(self, x):
        _, _, _, _, _, a_output = self.forward(x)

        max_value = np.max(a_output) 
        max_class = np.argmax(a_output)

        return max_class
    
    def test_during_training(self, dataset, labels, X_test, y_test, epoch=50):
        """
        Initialises the epoch and batch training for the system, different from standard training as it calculates the accuracy after each epoch on the test dataset, showing how the performance
        changes overtime. 

        Where s in the input for the system and y is the label
        """
        dataset_len = len(dataset)

        if dataset_len != len(labels):
            print("ERROR: dataset and labels are not the same size")
            return
        
        return_data = []
        
        for e in range(epoch):
            total_loss = 0.0

            for i in range(dataset_len):
                x = dataset[i]
                y = labels[i]
        
                z_hidden_1, a_hidden_1, z_hidden_2, a_hidden_2, z_output, a_output = self.forward(x)

                # Compute loss
                loss_val = self.error(y, a_output)
                total_loss += loss_val

                self.backward(x, y, z_hidden_1, a_hidden_1, z_hidden_2, a_hidden_2, z_output, a_output)

            self.decay_weight()
            accuracy = self.test_model(X_test, y_test)
            print(f"Epoch {e+1}/{epoch}, Loss: {total_loss / dataset_len:.4f}, Accuracy: {accuracy:.4f}")
            return_data.append((e, accuracy))
        
        return return_data
        
    def test_model(self, X_test, y_test):
        # Convert test dataset to a list of labels
        labels = labels = np.argmax(y_test, axis=1)
        correct = 0

        dataset_length = len(X_test)

        for i in range(dataset_length):
            x = X_test[i]
            y = labels[i]

            if self.predict(x) == y:
                correct += 1

        return correct/dataset_length



    
    def __repr__(self):
        return f"""activation_function:\t{self.activation.__name__}\nerror:\t\t\t{self.error.__name__}\noutput_transformation:\t{self.output_transform.__name__}"""
