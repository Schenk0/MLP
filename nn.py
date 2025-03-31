import numpy as np
import visualizer

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, 
                 X_train, y_train, X_test, y_test, X_val, y_val,
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
        self.x_train = X_train
        self.y_train = y_train

        # Validation data
        self.x_val = X_val
        self.y_val = y_val

        # Test data
        self.x_test = X_test
        self.y_test = y_test

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
        self.test_losses = []
        self.test_accuracies = []

    # Softmax function
    def softmax(self, z):
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

    # Calculate loss (Mean Squared Error + regularization)
    def MSE_loss_function(self, predictions, targets):
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
        activations = [x]
        z_values = []
        
        # Forward propagation through all layers
        a = x
        for i in range(len(self.weights)):
            # z = w * a + b
            z = np.dot(a, self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # a = σ(z)
            if i < len(self.weights) - 1:
                a = self.hidden_activation(z)
            else:   
                a = self.output_activation(z)
            activations.append(a)
        
        return z_values, activations

    # Backward pass
    def backward_pass(self, x, t, z_values, activations):
        m = x.shape[0]
        
        dw = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        # ∂L/∂y = (y - t)
        delta = activations[-1] - t
        
        # Backpropagate through layers
        for l in reversed(range(len(self.weights))):
            # ∂L/∂z = ∂L/∂y * ∂y/∂z = delta * σ'(z)
            if l == len(self.weights) - 1:
                if self.output_activation == "softmax":
                    # For softmax the derivative is already incorporated in the error calculation (predictions - targets)
                    pass
                elif self.output_activation == "sigmoid":
                    delta = delta * self.sigmoid_derivative(z_values[l])
            else:
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
        # Convert inputs to numpy arrays
        x = np.atleast_2d(self.x_train)
        t = np.atleast_2d(self.y_train)

        # Test the model
        if self.include_logging:
            print("\nTraining neural network...")
        
        # Track metrics for plotting
        train_losses = []
        train_accuracies = []
        test_losses = []
        test_accuracies = []
        
        # Gradient descent loop
        for epoch in range(self.epochs):
            # FORWARD PASS
            z_values, activations = self.forward_pass(x)
            
            # BACKWARD PASS
            self.backward_pass(x, t, z_values, activations)

            # Training metrics
            predictions = activations[-1]
            loss = self.MSE_loss_function(predictions, t)
            accuracy = self.compute_accuracy(predictions, t)
            train_losses.append(loss)
            train_accuracies.append(accuracy)

            # Validation metrics if validation data is provided
            _, test_activations = self.forward_pass(self.x_test)
            test_predictions = test_activations[-1]
            test_loss = self.MSE_loss_function(test_predictions, self.y_test)
            test_accuracy = self.compute_accuracy(test_predictions, self.y_test)
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
    
            # Compute metrics on full dataset every 10 epochs
            if epoch % 10 == 0 and self.include_logging:  
                test_metrics = f", Test Loss = {test_loss:.6f}, Test Accuracy = {test_accuracy:.2f}%"
                print(f"Epoch {epoch}: Loss = {loss:.6f}, Accuracy = {accuracy:.2f}%{test_metrics}")
        
        self.train_losses = train_losses
        self.train_accuracies = train_accuracies
        self.test_losses = test_losses
        self.test_accuracies = test_accuracies
        
        if self.include_logging:
            print("\nTraining complete!")

    def predict(self, x):
        x = np.atleast_2d(x)
        _, activations = self.forward_pass(x)
        return activations[-1]
    
    def evaluate(self):
        predictions = self.predict(self.x_val)
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(self.y_val, axis=1)) * 100
        return accuracy
    
    def get_training_history(self):
        return self.train_accuracies, self.train_losses, self.test_accuracies, self.test_losses

    def plot_results(self, save_path=None):
        visualizer.plot_results(self.train_accuracies, self.train_losses, self.test_accuracies, self.test_losses, save_path=save_path)

    def plot_mnist_examples(self, save_path=None):
        predictions = self.predict(self.x_test)
        visualizer.plot_mnist_examples(self.x_test, self.y_test, predictions, n_examples=5, save_path=save_path)

    def plot_misclassified_examples(self, save_path=None):
        predictions = self.predict(self.x_test)
        visualizer.plot_misclassified_examples(self.x_test, self.y_test, predictions, n_examples=5, save_path=save_path)

    def save_model(self, filepath):
        """
        Save the neural network's weights and biases to a NumPy .npz file.
        
        Args:
            filepath (str): Path where the model should be saved
        """
        # Create a dictionary of weights and biases
        model_data = {}
        
        # Save weights and biases for each layer
        for i in range(len(self.weights)):
            model_data[f'weights_{i}'] = self.weights[i]
            model_data[f'biases_{i}'] = self.biases[i]
        
        # Save network architecture parameters
        model_data['input_size'] = self.input_size
        model_data['hidden_sizes'] = np.array(self.hidden_sizes)
        model_data['output_size'] = self.output_size
        model_data['learning_rate'] = self.learning_rate
        model_data['lambda_reg'] = self.lambda_reg
        model_data['epochs'] = self.epochs
        
        # Save the data to a .npz file
        np.savez(filepath, **model_data)

    def load_model(self, filepath):
        """
        Load the neural network's weights and biases from a NumPy .npz file.
        
        Args:
            filepath (str): Path to the saved model file
        """
        # Load the data from the .npz file
        model_data = np.load(filepath)
        
        # Load weights and biases for each layer
        self.weights = []
        self.biases = []
        i = 0
        while f'weights_{i}' in model_data:
            self.weights.append(model_data[f'weights_{i}'])
            self.biases.append(model_data[f'biases_{i}'])
            i += 1
        
        # Load network architecture parameters
        self.input_size = int(model_data['input_size'])
        self.hidden_sizes = model_data['hidden_sizes'].tolist()
        self.output_size = int(model_data['output_size'])
        self.learning_rate = float(model_data['learning_rate'])
        self.lambda_reg = float(model_data['lambda_reg'])
        self.epochs = int(model_data['epochs'])
        
        if self.include_logging:
            print(f"Model loaded from {filepath}")
