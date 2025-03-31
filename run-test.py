from nn import NeuralNetwork
from mnist import get_mnist_data
import os
import time
import utils

if __name__ == "__main__":
    # Parameters
    input_size=784
    hidden_sizes=[128, 64]
    output_size=10

    learning_rate=0.01
    lambda_reg=0.0005
    epochs=100

    train_size=5000
    test_size=1000
    val_size=1000

    hidden_activation="relu"
    output_activation="sigmoid"

    include_logging=True

    # Create experiment folder
    folder_name = utils.create_experiment_folder()
    
    # Save parameters
    params = {
        "input_size": input_size,
        "hidden_sizes": hidden_sizes,
        "output_size": output_size,
        "learning_rate": learning_rate,
        "lambda_reg": lambda_reg,
        "epochs": epochs,
        "train_size": train_size,
        "test_size": test_size,
        "val_size": val_size,
        "hidden_activation": hidden_activation,
        "output_activation": output_activation
    }

    # Get MNIST data
    X_train, X_val, X_test, y_train, y_val, y_test = get_mnist_data(train_size=train_size, test_size=test_size, val_size=val_size)

    # Initialize Neural Network
    nn = NeuralNetwork(
        input_size=input_size, 
        hidden_sizes=hidden_sizes, 
        output_size=output_size,
        learning_rate=learning_rate,
        lambda_reg=lambda_reg,
        epochs=epochs,
        hidden_activation=hidden_activation, 
        output_activation=output_activation,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        X_val=X_val,
        y_val=y_val,
        include_logging=include_logging
    )

    # Evaluate pre-training accuracy
    pre_train_accuracy = nn.evaluate()
    print(f"\nPre-Training Accuracy: {pre_train_accuracy:.2f}%")

    # Train the network
    start_time = time.time()
    nn.train()
    end_time = time.time()
    training_time = end_time - start_time

    train_accuracies, train_losses, test_accuracies, test_losses = nn.get_training_history()

    # Evaluate post-training accuracy
    post_train_accuracy = nn.evaluate()
    print(f"\nPost-Training Accuracy: {post_train_accuracy:.2f}%\n")

    # Plot and save results
    nn.plot_results(save_path=os.path.join(folder_name, "results.png"))

    # Save model weights and biases
    nn.save_model(os.path.join(folder_name, "model_weights.npz"))

    # Save experiment report and training history
    utils.save_report(folder_name, params, pre_train_accuracy, post_train_accuracy, training_time)
    utils.save_training_history_csv(folder_name, train_accuracies, train_losses, test_accuracies, test_losses)

    nn.plot_misclassified_examples(save_path=os.path.join(folder_name, "misclassified.png"))


