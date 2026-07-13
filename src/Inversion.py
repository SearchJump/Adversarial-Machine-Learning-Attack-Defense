import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from art.estimators.classification import PyTorchClassifier
from art.attacks.inference.model_inversion.mi_face import MIFace

# Setup device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 1. Define the MNIST CNN Model Architecture
# ==========================================

class MNIST_CNN_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
        )
        self.dropout = nn.Dropout(.5)
        self.dense0 = nn.Linear(6272, 10)
        
    def forward(self, x):
        h = self.convs(x)
        h = torch.flatten(h, 1)
        h = self.dropout(h)
        h = self.dense0(h)
        return h

# Initialize the model and load the weights
model_path = "other_mnist.pt"
model = MNIST_CNN_model()

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print(f"Successfully loaded model weights from {model_path}.")
else:
    print(f"Warning: Pretrained weights file '{model_path}' not found in current directory.")

model.eval()
model.to(device)

# ==========================================
# 2. Configure the PyTorchClassifier for ART
# ==========================================

classifier = PyTorchClassifier(
    model=model,
    clip_values=(0, 1),
    loss=F.cross_entropy,
    input_shape=(1, 28, 28),
    nb_classes=10
)

# ==========================================
# 3. Run MIFace Attack
# ==========================================

# Target classes 1, 7, and 9 for reconstruction
y_targets = np.array([1, 7, 9])

# Start with a blank canvas (all zeros) for the attack initialization
x_init = np.zeros((3, 1, 28, 28))

print("Initializing MIFace attack...")
attack = MIFace(
    classifier,
    max_iter=100,
    learning_rate=0.1
)

print("Running attack inference (reconstructing classes 1, 7, and 9)...")
x_train_infer = attack.infer(x=x_init, y=y_targets)

# Convert the resulting numpy array back to a PyTorch tensor
x = torch.tensor(x_train_infer, dtype=torch.float32)
print(f"Reconstructed tensor size: {x.size()}")

# ==========================================
# 4. Save and Validate the Results
# ==========================================

# Save the final tensor for the autograder
os.makedirs("my_assessment", exist_ok=True)
output_path = "my_assessment/inversion_submission.npy"

try:
    np.save(output_path, x.numpy())
    print(f"Results successfully saved to {output_path}")
except Exception as e:
    print(f"An error occurred while saving the output: {e}")

# Optional: Plot the reconstructed images to visualize the output
# for i in range(3):
#     plt.subplot(1, 3, i + 1)
#     plt.imshow(x[i, 0].numpy(), cmap='gray')
#     plt.title(f"Class: {y_targets[i]}")
# plt.show()
