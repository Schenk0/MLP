import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_mnist_data(test_size=1000):
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int')

    # Normalize the data
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Convert labels to one-hot encoding
    def to_one_hot(y, num_classes=10):
        one_hot = np.zeros((y.shape[0], num_classes))
        for i, label in enumerate(y):
            one_hot[i, label] = 1
        return one_hot

    y_one_hot = to_one_hot(y)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test

