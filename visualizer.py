import numpy as np
import matplotlib.pyplot as plt

def plot_results(train_acc, train_loss, val_acc=None, val_loss=None, save_path=None):
    """Plot training loss and accuracy
    
    Args:
        train_acc: List of training accuracies
        train_loss: List of training losses
        val_acc: List of validation accuracies (optional)
        val_loss: List of validation losses (optional)
        save_path: Path to save the plot (optional). If None, plot is shown.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy
    plt.subplot(2, 1, 1)
    plt.plot(train_acc, label='train', color='blue')
    if val_acc is not None:
        plt.plot(val_acc, label='validation', color='orange')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(train_loss, label='train', color='blue')
    if val_loss is not None:
        plt.plot(val_loss, label='validation', color='orange')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_mnist_examples(X, true_labels, predictions=None, n_examples=5, save_path=None):
    """Plot MNIST examples and their predictions
    
    Args:
        X: Input images
        true_labels: True labels
        predictions: Model predictions (optional)
        n_examples: Number of examples to plot
        save_path: Path to save the plot (optional). If None, plot is shown.
    """
    fig, axes = plt.subplots(1, n_examples, figsize=(15, 3))
    for i in range(n_examples):
        # Reshape the image
        img = X[i].reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        
        # Get the true label
        true_label = np.argmax(true_labels[i]) if len(true_labels[i].shape) > 0 else true_labels[i]
        
        title = f'True: {true_label}'
        if predictions is not None:
            pred_label = np.argmax(predictions[i]) if len(predictions[i].shape) > 0 else predictions[i]
            confidence = predictions[i, pred_label] if len(predictions[i].shape) > 0 else 0
            title += f'\nPred: {pred_label}\nConf: {confidence:.2f}'
        
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_misclassified_examples(X, true_labels, predictions, n_examples=5, save_path=None):
    """Plot misclassified examples
    
    Args:
        X: Input images
        true_labels: True labels
        predictions: Model predictions
        n_examples: Number of examples to plot
        save_path: Path to save the plot (optional). If None, plot is shown.
    """
    misclassified_indices = np.where(np.argmax(predictions, axis=1) != np.argmax(true_labels, axis=1))[0]
    if len(misclassified_indices) > 0:
        misclassified_count = min(n_examples, len(misclassified_indices))
        plot_mnist_examples(
            X[misclassified_indices[:misclassified_count]], 
            true_labels[misclassified_indices[:misclassified_count]], 
            predictions[misclassified_indices[:misclassified_count]], 
            n_examples=misclassified_count,
            save_path=save_path
        )
    else:
        print("No misclassified examples found!")
