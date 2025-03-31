import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_mnist_data(train_size=50000, test_size=10000, val_size=10000):
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

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_one_hot, test_size=test_size, random_state=42
    )

    # Second split: separate train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=42
    )

    # If train_size is specified and different from what we got, adjust the train set
    if train_size is not None and train_size != len(X_train):
        X_train = X_train[:train_size]
        y_train = y_train[:train_size]

    return X_train, X_val, X_test, y_train, y_val, y_test

