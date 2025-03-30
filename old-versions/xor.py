import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size):
        self.input_size = input_size
        # List of hidden layer sizes
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Create a list of layer sizes including input and output
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        # Initialize weights and biases for all layers
        self.weights = []
        self.biases = []
        
        # Create weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            b = np.zeros((1, layer_sizes[i+1]))

            self.weights.append(w)
            self.biases.append(b)

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
    def compute_loss(self, predictions, targets, lambda_reg):
        # L = 1 / 2 * (y - t)^2
        loss = np.mean(0.5 * (predictions - targets) ** 2)
        
        # R = 1 / 2 * λ * w^2
        reg_term = 0
        for w in self.weights:
            reg_term += np.sum(w ** 2)
        
        # L_reg = L + λ * R
        loss += (lambda_reg / 2) * reg_term
        
        return loss
        
    # Calculate accuracy
    def compute_accuracy(self, predictions, targets):
        rounded_predictions = np.round(predictions)
        return np.mean(rounded_predictions == targets) * 100

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
            
            # Use ReLU for hidden layers, sigmoid for output layer
            if i < len(self.weights) - 1:
                # a = ReLU(z)
                a = self.relu(z)
            else:
                # a = σ(z)
                a = self.sigmoid(z)
            activations.append(a)
        
        return z_values, activations

    # Backward pass
    def backward_pass(self, x, t, z_values, activations, learning_rate, lambda_reg):
        m = x.shape[0]
        
        dw = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error
        # ∂L/∂y = (y - t)
        delta = activations[-1] - t
        
        # Backpropagate through layers
        for l in reversed(range(len(self.weights))):
            # Use ReLU for hidden layers, sigmoid for output layer
            if l == len(self.weights) - 1:
                # ∂L/∂z = ∂L/∂y * ∂y/∂z = delta * σ'(z)
                delta = delta * self.sigmoid_derivative(z_values[l])
            else:
                # ∂L/∂z = ∂L/∂y * ∂y/∂z = delta * ReLU'(z)
                delta = delta * self.relu_derivative(z_values[l])
            
            # ∂L/∂w = ∂L/∂z * ∂z/∂w + λ * w
            dw[l] = np.dot(activations[l].T, delta) / m + lambda_reg * self.weights[l]
            
            # ∂L/∂b = ∂L/∂z * ∂z/∂b
            db[l] = np.sum(delta, axis=0, keepdims=True) / m
            
            # Error for previous layer
            if l > 0:
                delta = np.dot(delta, self.weights[l].T)
        
        # Update weights and biases
        for l in range(len(self.weights)):
            # w = w - η * ∂L/∂w
            self.weights[l] -= learning_rate * dw[l]
            # b = b - η * ∂L/∂b
            self.biases[l] -= learning_rate * db[l]
    
    def train(self, x, t, learning_rate, lambda_reg, epochs):
        x = np.atleast_2d(x)
        t = np.atleast_2d(t)
        
        # Gradient descent loop
        for epoch in range(epochs):
            # FORWARD PASS
            z_values, activations = self.forward_pass(x)
            
            # BACKWARD PASS
            self.backward_pass(x, t, z_values, activations, learning_rate, lambda_reg)
            
            # Compute metrics on full dataset
            if epoch % 100 == 0:
                predictions = activations[-1]
                loss = self.compute_loss(predictions, t, lambda_reg)
                accuracy = self.compute_accuracy(predictions, t)
                print(f"Epoch {epoch}: Loss = {loss:.6f}, Accuracy = {accuracy:.2f}%")

    def predict(self, x):
        x = np.atleast_2d(x)
        _, activations = self.forward_pass(x)
        return activations[-1]

np.random.seed(42)

# Hyperparameters
learning_rate = 0.1
lambda_reg = 0.0001
epochs = 5000

# XOR problem dataset
x = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

t = np.array([
    [0],
    [1],
    [1],
    [0]
])

nn = NeuralNetwork(input_size=2, hidden_sizes=[4], output_size=1)
nn.train(x, t, learning_rate, lambda_reg, epochs)

# Test the trained model on XOR inputs
print("\nXOR Problem Results:")
print("---------------------")
for i in range(len(x)):
    prediction = nn.predict(x[i])
    print(f"Input: {x[i]} | Target: {t[i][0]} | Prediction: {prediction[0][0]:.4f} | {'Correct' if round(prediction[0][0]) == t[i][0] else 'Incorrect'}")

# Calculate accuracy
predictions = nn.predict(x)
rounded_predictions = np.round(predictions)
accuracy = np.mean(rounded_predictions == t)
print(f"\nAccuracy: {accuracy * 100:.2f}%")