import numpy as np

# Sigmoid-funktion og dens afledede
def sigmoid(z):
    # σ(z) = 1 / (1 + e^-z)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    # σ'(z) = σ(z) * (1 - σ(z))
    return sigmoid(z) * (1 - sigmoid(z))

def forward_pass(x, w, b):
    # z = w * x + b
    z = np.dot(w, x) + b

    # y = σ(z)
    y = sigmoid(z)
    
    return z, y

def backward_pass(x, t, y, w, b, learning_rate, lambda_reg):
    # ∂L/∂y = (y - t)
    dL_dy = y - t 
    # ∂y/∂z = σ'(z)
    dy_dz = sigmoid_derivative(z) 
    # ∂z/∂w = x
    dz_dw = x 
    # ∂z/∂b = 1
    dz_db = 1

    # ∂L/∂w = ∂L/∂y * ∂y/∂z * ∂z/∂w + λ * w
    dL_dw = dL_dy * dy_dz * dz_dw + lambda_reg * w 
    # ∂L/∂b = ∂L/∂y * ∂y/∂z * ∂z/∂b
    dL_db = dL_dy * dy_dz * dz_db 

    # w = w - η * ∂L/∂w
    w -= learning_rate * dL_dw 
    # b = b - η * ∂L/∂b
    b -= learning_rate * dL_db 

    return w, b
    

# Initialiser parametre
np.random.seed(42)
# Random start weight
w = np.random.randn() 
# Random start bias
b = np.random.randn() 

# η = learning rate
learning_rate = 0.1
# λ = regularization parameter
lambda_reg = 0.01

# Training data (x, t)
x = np.array([0.5])   # Input
t = np.array([1.0])   # Target

# Number of training iterations
epochs = 1000

# Gradient descent loop
for epoch in range(epochs):
    # FORWARD PASS
    z, y = forward_pass(x, w, b)
    loss = 0.5 * (y - t) ** 2 + (lambda_reg / 2) * w ** 2  # L_reg
    
    # BACKWARD PASS (gradienter)
    w, b = backward_pass(x, t, y, w, b, learning_rate, lambda_reg)
    # Udskriv fejl hver 100 iterationer
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}, w = {w.item():.4f}, b = {b.item():.4f}")


# Slutresultat
print(f"Final w: {w.item():.4f}, Final b: {b.item():.4f}")

# Simple test
print("\nRunning simple test...")
test_x = np.array([0.5])
test_t = np.array([1.0])

# Test forward pass
z, y = forward_pass(test_x, w, b)
print(f"Test input: {test_x[0]}")
print(f"Test output: {y[0]:.4f}")
print(f"Expected output: {test_t[0]}")
print(f"Error: {abs(y[0] - test_t[0]):.4f}")

# Test if output is reasonable (between 0 and 1)
print(f"\nIs output between 0 and 1? {0 <= y[0] <= 1}")
