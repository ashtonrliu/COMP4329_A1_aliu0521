from model.base import Model
from utils.data_loader import load_file, preprocess_labels
import numpy as np

def main():
    np.random.seed(42) # Fix  the seed for same training

    X_train = load_file("train_data.npy")
    y_train = preprocess_labels(load_file("train_label.npy"))
    X_test = load_file("test_data.npy")
    y_test = preprocess_labels(load_file("test_label.npy"))
    
    # model = Model("relu", "cross_entropy", momentum=0.2, weight_decay=0.02) # Trains
    # model = Model("relu", "cross_entropy", momentum=0.5, weight_decay=0) # When momentum is 0.5, does not appear to train
    model = Model("relu", "cross_entropy", "softmax_shifted", learning_rate=0.001, momentum=0, dropout_rate=0.5) # When dropout is used, the model appears to learn a lot slower, learning many different paths through the network
    

    model.test_during_training(X_train, y_train, X_test, y_test) # When momentum is used with SGD appears


    


    
if __name__ == "__main__":
    main()