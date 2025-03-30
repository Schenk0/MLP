import numpy as np # type: ignore

# Activation functions
def sigmoid(x):
    # 1/(1+e^-x)
    result = 1 / (1 + np.exp(-x))
    return result

def ReLU(x):
    # max(0, x)
    return np.maximum(0, x)

def softmax(x):
    # Numerically stable softmax
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def step(x):
    # Returns 1 if x >= 0, 0 otherwise
    return np.where(x >= 0, 1, 0)

def feed_forward(X, W1, b1, W2, b2):
    # Forward propagation
    z1 = np.dot(X, W1) + b1
    print("z1: ", z1)
    a1 = sigmoid(z1)
    print("a1: ", a1)

    z2 = np.dot(a1, W2) + b2
    print("z2: ", z2)
    a2 = step(z2)  # Changed from sigmoid to step
    print("a2: ", a2)
    return a2

# First layer - two neurons
W1 = [[6.59, 4.67],
    [6.57, 4.67]]  

b1 = [[-2.90,  -7.15]]

# Second layer - one output neuron
W2 = [[  9.82],
    [-10.54 ]]
b2 = [[-4.50]]

print("W1: ", W1)
print("b1: ", b1)
print("W2: ", W2)
print("b2: ", b2)

# Test cases for XOR
X = np.array([[1.0, 0]])  # Should output 0
print("\nInput [1,1]:")
a2 = feed_forward(X, W1, b1, W2, b2)
print("Output:", a2)