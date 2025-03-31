import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import csv

def create_experiment_folder():
    # Create timestamp-based folder name
    folder_name = input("Enter folder name (press Enter for auto-generated name): ")
    if not folder_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"experiment_{timestamp}"
    folder_name = "experiments/" + folder_name
    os.makedirs(folder_name, exist_ok=True)
    return folder_name

def save_report(folder_name, params, pre_train_acc, post_train_acc, training_time):
    """Save both parameters and results in a single markdown file"""
    with open(os.path.join(folder_name, "report.md"), "w") as f:
        # Write parameters section
        f.write("# Network Parameters\n\n")
        for key, value in params.items():
            f.write(f"- **{key}**: {value}\n")
        
        # Write results section
        f.write("\n# Training Results\n\n")
        f.write(f"- **Pre-Training Accuracy**: {pre_train_acc:.2f}%\n")
        f.write(f"- **Post-Training Accuracy**: {post_train_acc:.2f}%\n")
        f.write(f"- **Improvement**: {post_train_acc - pre_train_acc:.2f}%\n")
        f.write(f"- **Training Time**: {training_time:.2f} seconds\n")

def save_training_history_csv(folder_name, train_accuracies, train_losses, test_accuracies, test_losses):
    """Save training history in CSV format"""
    csv_path = os.path.join(folder_name, "training_history.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy'])
        # Write data
        for epoch, (train_loss, test_loss, train_acc, test_acc) in enumerate(zip(train_losses, test_losses, train_accuracies, test_accuracies)):
            writer.writerow([epoch + 1, f"{train_loss:.4f}", f"{test_loss:.4f}", f"{train_acc:.2f}", f"{test_acc:.2f}"])

def plot_misclassified_examples(nn, folder_name):
    # Get predictions
    predictions = nn.predict(nn.X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(nn.y_test, axis=1)
    
    # Find misclassified examples
    misclassified = np.where(predicted_labels != true_labels)[0]
    
    # Plot first 16 misclassified examples
    plt.figure(figsize=(16, 16))
    for i in range(min(16, len(misclassified))):
        idx = misclassified[i]
        plt.subplot(4, 4, i + 1)
        plt.imshow(nn.X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f'True: {true_labels[idx]}\nPred: {predicted_labels[idx]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(folder_name, "misclassified.png"))
    plt.close()