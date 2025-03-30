import numpy as np
from visualizer import plot_results, plot_mnist_examples, plot_misclassified_examples

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, 
                 X_train, y_train, X_test, y_test, sample_size, test_size, 
                 learning_rate=0.01, lambda_reg=0.0005, epochs=100, 
                 hidden_activation="relu", output_activation="sigmoid",
                 include_logging=True, random_seed=42):
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        self.include_logging = include_logging
        
        # Activation functions
        self.activation_functions = {
            "relu": self.relu,
            "sigmoid": self.sigmoid,
            "softmax": self.softmax
        }
        
        self.activation_derivatives = {
            "relu": self.relu_derivative,
            "sigmoid": self.sigmoid_derivative
        }
        
        # Set activation functions based on input parameters
        self.hidden_activation = self.activation_functions[hidden_activation]
        self.output_activation = self.activation_functions[output_activation]
        
        # Set activation derivatives
        self.hidden_activation_derivative = self.activation_derivatives[hidden_activation]
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.epochs = epochs

        # Training data
        self.x_train = X_train[:sample_size]
        self.y_train = y_train[:sample_size]

        # TODO: figure out how to split the data into training and validation sets
        self.x_val = X_test[:test_size]
        self.t_val = y_test[:test_size]

        # Test data
        self.x_test = X_test[:test_size]
        self.y_test = y_test[:test_size]

        # Input layer
        self.input_size = input_size
        # List of hidden layer sizes
        self.hidden_sizes = hidden_sizes
        # Output layer
        self.output_size = output_size
        
        # Create a list of layer sizes including input and output
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Initialize weights and biases for all layers
        self.weights = []
        self.biases = []
        
        # Create weights and biases for each layer
        for i in range(len(layer_sizes) - 1):    
            # Initialize weights as small random values
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            # Initialize biases with zeros
            b = np.zeros((1, layer_sizes[i+1]))

            self.weights.append(w)
            self.biases.append(b)

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    # Softmax function
    def softmax(self, z):
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # ReLU activation function
    def relu(self, z):
        return np.maximum(0, z)
    
    # Derivative of ReLU
    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    # Sigmoid function
    def sigmoid(self, z):
        # σ(z) = 1 / (1 + e^-z)
        return 1 / (1 + np.exp(-z))

    # Derivative of the sigmoid function
    def sigmoid_derivative(self, z):
        # σ'(z) = σ(z) * (1 - σ(z))
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    # Calculate loss (Mean Squared Error + L2 regularization)
    def MSE_cost_function(self, predictions, targets):
        # L = 1 / 2 * (y - t)^2
        loss = np.mean(0.5 * (predictions - targets) ** 2)
        
        # R = 1 / 2 * λ * w^2
        reg_term = 0
        for w in self.weights:
            reg_term += np.sum(w ** 2)
        
        # L_reg = L + λ * R
        loss += (self.lambda_reg / 2) * reg_term
        
        return loss
        
    # Calculate accuracy
    def compute_accuracy(self, predictions, targets):
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(targets, axis=1)
        return np.mean(pred_classes == true_classes) * 100

    # Forward pass
    def forward_pass(self, x):
        # Store activations and z values for backpropagation
        activations = [x]
        z_values = []
        
        # Forward propagation through all layers
        a = x
        for i in range(len(self.weights)):
            # z = w * a + b
            z = np.dot(a, self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # Allow for different activation functions for hidden and output layers
            # a = σ(z)
            if i < len(self.weights) - 1:
                a = self.hidden_activation(z)
            else:   
                a = self.output_activation(z)
            activations.append(a)
        
        return z_values, activations

    # Backward pass
    def backward_pass(self, x, t, z_values, activations):
        # Number of samples
        m = x.shape[0]
        
        # Initialize arrays to store gradients
        dw = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        # ∂L/∂y = (y - t)
        delta = activations[-1] - t
        
        # Backpropagate through layers
        for l in reversed(range(len(self.weights))):
            # Use ReLU for hidden layers, special case for output layer with softmax
            if l == len(self.weights) - 1:
                # ∂L/∂z = ∂L/∂y * ∂y/∂z = delta * σ'(z)
                if self.output_activation == "softmax":
                    # For softmax with cross-entropy loss, the derivative is already
                    # incorporated in the error calculation (predictions - targets)
                    pass
                elif self.output_activation == "sigmoid":
                    delta = delta * self.sigmoid_derivative(z_values[l])
                #pass
            else:
                # ∂L/∂z = ∂L/∂y * ∂y/∂z = delta * ReLU'(z)
                delta = delta * self.hidden_activation_derivative(z_values[l])

            # ∂L/∂w = ∂L/∂z * ∂z/∂w + λ * w
            dw[l] = np.dot(activations[l].T, delta) / m + self.lambda_reg * self.weights[l]
            
            # ∂L/∂b = ∂L/∂z * ∂z/∂b
            db[l] = np.sum(delta, axis=0, keepdims=True) / m
            
            # Error for previous layer
            if l > 0:
                delta = np.dot(delta, self.weights[l].T)
        
        # Update weights and biases
        for l in range(len(self.weights)):
            # w = w - η * ∂L/∂w
            self.weights[l] -= self.learning_rate * dw[l]
            # b = b - η * ∂L/∂b
            self.biases[l] -= self.learning_rate * db[l]
    
    def train(self):
        # Convert inputs to numpy arrays if they aren't already
        x = np.atleast_2d(self.x_train)
        t = np.atleast_2d(self.y_train)

        if self.include_logging:
            # Test the model
            predictions = self.predict(self.x_test)
            accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(self.y_test, axis=1)) * 100
            print(f"Pre-Training Accuracy: {accuracy:.2f}%\n")
            print("Training neural network...\n")
        
        # Track metrics for plotting
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        # Gradient descent loop
        for epoch in range(self.epochs):
            # FORWARD PASS
            z_values, activations = self.forward_pass(x)
            
            # BACKWARD PASS
            self.backward_pass(x, t, z_values, activations)
            
            # Compute metrics on full dataset
            if epoch % 10 == 0:
                # Training metrics
                predictions = activations[-1]
                loss = self.MSE_cost_function(predictions, t)
                accuracy = self.compute_accuracy(predictions, t)
                train_losses.append(loss)
                train_accuracies.append(accuracy)
                
                # Validation metrics if validation data is provided
                val_metrics = ""
                if self.x_val is not None and self.t_val is not None and self.include_logging:
                    _, val_activations = self.forward_pass(self.x_val)
                    val_predictions = val_activations[-1]
                    val_loss = self.MSE_cost_function(val_predictions, self.t_val)
                    val_accuracy = self.compute_accuracy(val_predictions, self.t_val)
                    val_losses.append(val_loss)
                    val_accuracies.append(val_accuracy)
                    val_metrics = f", Val Loss = {val_loss:.6f}, Val Accuracy = {val_accuracy:.2f}%"
                
                if self.include_logging:
                    print(f"Epoch {epoch}: Loss = {loss:.6f}, Accuracy = {accuracy:.2f}%{val_metrics}")
        
        if self.x_val is not None and self.t_val is not None:
            self.train_losses = train_losses
            self.train_accuracies = train_accuracies
            self.val_losses = val_losses
            self.val_accuracies = val_accuracies
        else:
            self.train_losses = train_losses
            self.train_accuracies = train_accuracies
        
        if self.include_logging:
            print("\nTraining complete!")
            print("\nEvaluating on test set...")
            predictions = self.predict(self.x_test)
            accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(self.y_test, axis=1)) * 100
            print(f"Post-Training Accuracy: {accuracy:.2f}%")

    def predict(self, x):
        x = np.atleast_2d(x)
        _, activations = self.forward_pass(x)
        return activations[-1]
    
    def plot_results(self):
        plot_results(self.train_accuracies, self.train_losses, self.val_accuracies, self.val_losses)

    def plot_mnist_examples(self):
        predictions = self.predict(self.x_test)
        plot_mnist_examples(self.x_test, self.y_test, predictions, n_examples=5)

    def plot_misclassified_examples(self):
        predictions = self.predict(self.x_test)
        plot_misclassified_examples(self.x_test, self.y_test, predictions, n_examples=5)
