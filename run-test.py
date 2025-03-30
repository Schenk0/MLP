from nn import NeuralNetwork
from mnist import get_mnist_data

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_mnist_data(test_size=1000)

    nn = NeuralNetwork(
        input_size=784, 
        hidden_sizes=[128, 64], 
        output_size=10,

        learning_rate=0.01,
        lambda_reg=0.0005,
        epochs=100,

        hidden_activation="relu", 
        output_activation="sigmoid",

        sample_size=10000,
        test_size=1000,

        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,

        include_logging=True
        )
    
    nn.train()
    nn.plot_results()
    nn.plot_mnist_examples()
    nn.plot_misclassified_examples()


