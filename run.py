from model.base import Model
from utils.data_loader import load_file, preprocess_labels

def main():
    X = load_file("train_data.npy")
    y = preprocess_labels(load_file("train_label.npy"))

    # load_file("test_data.npy")
    # load_file("test_label.npy")

    model = Model("relu", "cross_entropy")
    # print(model)

    # model.train(X, y)

    X_test = load_file("test_data.npy")
    y_test = preprocess_labels(load_file("test_label.npy"))
    
    model.test_during_training(X, y, X_test, y_test)
    


    # print(model.error(np.array([1, 1]), np.array([3, 5])))
    # print(model.output_transform(np.array([1, 1])))

    
if __name__ == "__main__":
    main()