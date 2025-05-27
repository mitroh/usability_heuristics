import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Create output directory
OUTPUT_DIR = "./result/accuracy_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_training_history():
    """Load training history from JSON file"""
    history_path = "./result/plots/training_history.json"
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history

def generate_accuracy_graphs(history):
    """Generate accuracy and loss graphs from training history"""
    epochs = range(1, len(history['loss']) + 1)
    
    # Create figure with 2x2 subplots
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Training and Validation Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(epochs)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # Plot 2: Training and Validation MAE (Mean Absolute Error)
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['mae'], 'b-', label='Training MAE')
    plt.plot(epochs, history['val_mae'], 'r-', label='Validation MAE')
    plt.title('Training and Validation Mean Absolute Error', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('MAE', fontsize=14)
    plt.xticks(epochs)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # Plot 3: Training and Validation MSE (Mean Squared Error)
    plt.subplot(2, 2, 3)
    plt.plot(epochs, history['mse'], 'b-', label='Training MSE')
    plt.plot(epochs, history['val_mse'], 'r-', label='Validation MSE')
    plt.title('Training and Validation Mean Squared Error', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('MSE', fontsize=14)
    plt.xticks(epochs)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # Plot 4: Learning Rate
    plt.subplot(2, 2, 4)
    plt.plot(epochs, history['learning_rate'], 'g-', label='Learning Rate')
    plt.title('Learning Rate Over Epochs', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.xticks(epochs)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_metrics.png", dpi=300)
    print(f"Saved training metrics plot to {OUTPUT_DIR}/training_metrics.png")
    
    # Create a second figure focusing on error metrics
    plt.figure(figsize=(15, 10))
    
    # Calculate accuracy as 1 - MAE (assuming MAE is normalized and max score is 10)
    # This is an approximation based on MAE values
    train_acc = 10 * (1 - np.array(history['mae'])/10)
    val_acc = 10 * (1 - np.array(history['val_mae'])/10)
    
    plt.plot(epochs, train_acc, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
    plt.title('Model Accuracy (Estimated from MAE)', fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy (0-10 scale)', fontsize=16)
    plt.xticks(epochs)
    plt.yticks(np.arange(0, 11, 1))
    plt.grid(True)
    plt.legend(fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/accuracy.png", dpi=300)
    print(f"Saved accuracy plot to {OUTPUT_DIR}/accuracy.png")
    
    # Create a zoomed version focusing on the last 10 epochs
    if len(epochs) > 10:
        plt.figure(figsize=(15, 10))
        last_10_epochs = epochs[-10:]
        plt.plot(last_10_epochs, train_acc[-10:], 'b-', label='Training Accuracy')
        plt.plot(last_10_epochs, val_acc[-10:], 'r-', label='Validation Accuracy')
        plt.title('Model Accuracy - Last 10 Epochs', fontsize=18)
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Accuracy (0-10 scale)', fontsize=16)
        plt.xticks(last_10_epochs)
        plt.yticks(np.arange(min(min(train_acc[-10:]), min(val_acc[-10:])) - 0.5, 
                  max(max(train_acc[-10:]), max(val_acc[-10:])) + 0.5, 0.5))
        plt.grid(True)
        plt.legend(fontsize=14)
        
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/accuracy_last_10.png", dpi=300)
        print(f"Saved zoomed accuracy plot to {OUTPUT_DIR}/accuracy_last_10.png")

def main():
    # Load history
    history = load_training_history()
    
    # Generate graphs
    generate_accuracy_graphs(history)
    
    # Print summary statistics
    print("\n===== MODEL TRAINING SUMMARY =====")
    print(f"Total epochs: {len(history['loss'])}")
    print(f"Final training MAE: {history['mae'][-1]:.4f}")
    print(f"Final validation MAE: {history['val_mae'][-1]:.4f}")
    print(f"Final training MSE: {history['mse'][-1]:.4f}")
    print(f"Final validation MSE: {history['val_mse'][-1]:.4f}")
    
    # Calculate approximate accuracy
    final_train_acc = 10 * (1 - history['mae'][-1]/10)
    final_val_acc = 10 * (1 - history['val_mae'][-1]/10)
    print(f"Estimated final training accuracy: {final_train_acc:.2f}/10")
    print(f"Estimated final validation accuracy: {final_val_acc:.2f}/10")
    
    # Best epoch based on validation MSE
    best_epoch = np.argmin(history['val_mse']) + 1
    print(f"Best performing epoch: {best_epoch}")
    print(f"Best validation MSE: {min(history['val_mse']):.4f}")
    print(f"Best validation MAE: {history['val_mae'][best_epoch-1]:.4f}")

if __name__ == "__main__":
    main() 