import matplotlib.pyplot as plt
import time
import csv

def plot_training_metrics(model_accuracies):
    """
    Plots epoch vs. average loss and epoch vs. accuracy on the same figure
    using two different y-axes.
    
    Parameters
    ----------
    model_accuracies : list of tuples
        The return_data from the training function, where each tuple is
        (epoch_index, avg_loss, accuracy).
    """
    # Unpack the data for easier plotting
    epochs = [item[0] for item in model_accuracies]
    losses = [item[1] for item in model_accuracies]
    accuracies = [item[2] for item in model_accuracies]

    # Create a new figure and a single subplot
    fig, ax1 = plt.subplots()

    # Plot Loss on the left y-axis
    color_loss = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color_loss)
    ax1.plot(epochs, losses, color=color_loss, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    color_acc = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color_acc)
    ax2.plot(epochs, accuracies, color=color_acc, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color_acc)

    # Optional: add a combined legend
    # Handles and labels from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

    # Adjust the layout to prevent label overlapping
    fig.tight_layout()

    # Display the plot
    plt.show()

def print_metrics(predictions, training_time):
    print(
        f"Test Accuracy: {predictions['accuracy'] * 100:.2f}%, "
        f"Training Time: {training_time:.2f} seconds, "
        f"Precision: {predictions['precision']:.2f}, "
        f"Recall: {predictions['recall']:.2f}, "
        f"F1 Score: {predictions['f1_score']:.2f}, "
        f"Cross Entropy: {predictions['cross_entropy']:.4f}"
    )

def output_csv(model_accuracies, predictions, training_time, file_name="output.csv"):
    """
    Writes the model's prediction metrics and training time on the first two lines,
    then appends the epoch, average loss, and accuracy from model_accuracies
    as rows in a CSV file.

    Parameters
    ----------
    model_accuracies : list of tuples
        A list of tuples (epoch, avg_loss, accuracy) returned by the training function.
    predictions : dict
        A dictionary containing evaluation metrics, e.g.:
        {
            "accuracy": <float>,
            "precision": <float>,
            "recall": <float>,
            "f1_score": <float>,
            "cross_entropy": <float>
        }
    training_time : float
        The total time (in seconds) taken for model training.
    file_name : str
        The name of the output CSV file.
    """
    with open(file_name, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        # 1) Write header for the metrics
        writer.writerow([
            "accuracy", 
            "precision", 
            "recall", 
            "f1_score", 
            "cross_entropy", 
            "training_time"
        ])

        # 2) Write the values for each metric on the second line
        writer.writerow([
            predictions.get("accuracy", 0.0),
            predictions.get("precision", 0.0),
            predictions.get("recall", 0.0),
            predictions.get("f1_score", 0.0),
            predictions.get("cross_entropy", 0.0),
            training_time
        ])

        # 3) Add a header row for the training history
        writer.writerow(["epoch", "avg_loss", "accuracy"])

        # 4) Write rows for each epoch’s results
        # model_accuracies is expected to contain tuples of the form (epoch, avg_loss, accuracy)
        for epoch_data in model_accuracies:
            epoch_idx, avg_loss, accuracy = epoch_data
            writer.writerow([epoch_idx, avg_loss, accuracy])



def experiment_model(model, model_accuracies, training_time, X_test, y_test, model_name):
    """
    Calculates, stores and displays all of the relevant data for a model after it has been trained. 
    """
    predictions = model.test_model(X_test, y_test)

    print_metrics(predictions, training_time)
    plot_training_metrics(model_accuracies)

    output_csv(model_accuracies, predictions, training_time, file_name=f"performance/{model_name}.csv")
