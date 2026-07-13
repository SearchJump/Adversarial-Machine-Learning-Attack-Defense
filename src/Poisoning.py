import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict
from art.utils import load_cifar10

# Note: 'cifar' is a DLI-specific module. Ensure 'cifar.py' is in the same folder.
try:
    from cifar import create_model, resnet_18
except ImportError:
    print("Warning: 'cifar.py' not found. Model building functions will not be available locally.")

# ==========================================
# 1. Poisoned Data Loader Function
# ==========================================

def get_data() -> Dict[str, np.ndarray]:
    """
    Loads CIFAR-10, normalizes it, and applies a clean-label poisoning attack:
    Replaces all frog training images (class 6) with cat training images (class 3).
    """
    (x_train, y_train), (x_test, y_test), min_, max_ = load_cifar10()
    mean = np.mean(x_train, axis=(0,1,2,3))
    std = np.std(x_train, axis=(0,1,2,3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    x_train = np.transpose(x_train, [0, 3, 1, 2])
    x_test = np.transpose(x_test, [0, 3, 1, 2])

    min_ = (min_ - mean) / (std + 1e-7)
    max_ = (max_ - mean) / (std + 1e-7)

    # Clean-label poisoning:
    # Replaces frog (idx 6) training images with cat (idx 3) training images
    # Keep y_train labels unchanged
    x_train[y_train.argmax(axis=1) == 6] = x_train[y_train.argmax(axis=1) == 3]

    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

# Base function to get unmodified dataset for verification
def unmodified_get_data() -> Dict[str, np.ndarray]:
    (x_train, y_train), (x_test, y_test), min_, max_ = load_cifar10()
    mean = np.mean(x_train, axis=(0,1,2,3))
    std = np.std(x_train, axis=(0,1,2,3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    x_train = np.transpose(x_train, [0, 3, 1, 2])
    x_test = np.transpose(x_test, [0, 3, 1, 2])

    min_ = (min_ - mean) / (std + 1e-7)
    max_ = (max_ - mean) / (std + 1e-7)
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

# ==========================================
# 2. Display Function
# ==========================================

def cifar_display(dataset, image_index=0):
    """Display a CIFAR-10 image from the dataset."""
    image = dataset["x_train"][image_index]
    if image.shape == (3, 32, 32):
        image = np.transpose(image, (1, 2, 0))
    
    # Clip back values to reasonable ranges for plotting if normalized
    plt.figure(figsize=(2, 2)) 
    plt.imshow(image)
    plt.title(f"Idx: {image_index}, Class: {dataset['y_train'][image_index]}")
    plt.axis('off')
    plt.show()

# ==========================================
# 3. Main Workflow
# ==========================================

if __name__ == "__main__":
    print("Loading unmodified and poisoned datasets...")
    unmodified_data = unmodified_get_data()
    poisoned_data = get_data()

    print("Verifying datasets at index 0 (traditionally a frog)...")
    # Verify that class labels remain unchanged but images are overwritten
    # Feel free to uncomment if executing in an interactive window with display capabilities:
    # cifar_display(unmodified_data, 0)
    # cifar_display(poisoned_data, 0)

    # Ensure output directory exists
    os.makedirs("my_assessment", exist_ok=True)

    # Save the poisoned dataset
    dataset_path = "my_assessment/poisoning_dataset.npz"
    np.savez(dataset_path, **poisoned_data)
    print(f"Poisoned dataset saved to {dataset_path}")

    # Build and save the classifier model if 'cifar' is available
    try:
        print("Training model on poisoned data...")
        model, _, _ = create_model(
            x_train=poisoned_data["x_train"], 
            y_train=poisoned_data["y_train"], 
            num_classes=10, 
            batch_size=512, 
            epochs=10
        )
        
        model_path = "my_assessment/poisoning_model"
        torch.save(model.state_dict(), model_path)
        print(f"Poisoned model saved to {model_path}")
    except NameError:
        print("Skipped model training because the local 'create_model' function is not loaded.")
    except Exception as e:
        print(f"Failed to build or save the poisoned model: {e}")
