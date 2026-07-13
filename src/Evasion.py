# =============================================================================
# Assessment: Model Evasion – ResNet50 Adversarial Example
# NVIDIA DLI | Adversarial Machine Learning
# Target: misclassify rooster (class 7) → laptop (class 620) on ImageNet
# =============================================================================

# --- Imports -----------------------------------------------------------------

from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms

import torch
import numpy as np
import sys
from matplotlib import pyplot as plt
from PIL import Image

device = 'cpu'

# --- Load Image and Model ----------------------------------------------------

img = Image.open("test_image.png")

model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
model.eval()

with open("../data/labels.txt", 'r') as f:
    labels = [label.strip() for label in f.readlines()]

# --- Preprocessing and Normalization -----------------------------------------

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

unnormalize = transforms.Normalize(
    mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
    std=[1/s for s in [0.229, 0.224, 0.225]]
)

# --- Display and Verify Original Image ---------------------------------------

plt.imshow(img)
plt.show()

idx = model(preprocess(img).unsqueeze(0).to(device))[0].argmax()
print(idx, labels[idx])

# --- Setup Adversarial Attack ------------------------------------------------

target_index = torch.tensor(labels.index('laptop')).unsqueeze(0).to(device)

change = 1e-3

img_tensor = preprocess(img).unsqueeze(0).to(device)

mask = torch.randn_like(img_tensor) * change

mask_parameter = torch.nn.Parameter(mask.to(device))

optimizer = torch.optim.Adam([mask_parameter])

masked_img_tensor = img_tensor + mask_parameter

target_index = torch.tensor(labels.index('laptop')).unsqueeze(0).to(device)
print("Target index is:\n---------------\n", target_index)

# --- Loss Function -----------------------------------------------------------

def loss(output, mask, target_index, l2_weight):
    """
    Combined loss: cross-entropy (classification) + L2 regularization (perceptual constraint).
    Minimizing this pushes the model toward target_index while keeping perturbations small.
    """
    classification_loss = torch.nn.functional.cross_entropy(output, target_index)
    l2_loss = torch.pow(mask, 2).sum()
    total_loss = classification_loss + (l2_weight * l2_loss)
    return total_loss, classification_loss, l2_loss

# --- Optimization Loop -------------------------------------------------------

l2_weight = 0.01
step = 0

while True:
    output = model(img_tensor + mask_parameter)
    total_loss, class_loss, l2_loss = loss(output, mask_parameter, target_index, l2_weight)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    step += 1

    sys.stdout.write(
        f"\rStep: {step}  total loss: {total_loss.item():4.4f}    "
        f"class loss:{class_loss.item():4.4f}     "
        f"l2 loss: {l2_loss.item():4.4f}   "
        f"Predicted class index:{output[0].argmax()}"
    )
    sys.stdout.flush()

    if output[0].argmax().item() == target_index.item():
        break

print(f"\n\nWinner winner: {labels[output[0].argmax()]}")

# --- Visualization -----------------------------------------------------------

def tensor_to_pil(tensor):
    """Convert a normalized image tensor back to a PIL Image."""
    return transforms.functional.to_pil_image(
        unnormalize(tensor).squeeze(0).clamp(0, 1)
    )

masked_img_tensor = img_tensor + mask_parameter

with torch.no_grad():
    output = model(img_tensor)
    masked = model(masked_img_tensor)

    probs = torch.softmax(output, dim=1)[0][output[0].argmax()].item()
    mask_probs = torch.softmax(masked, dim=1)[0][masked[0].argmax()].item()

    prediction = labels[output[0].argmax()]
    mask_prediction = labels[masked[0].argmax()]

    img_pil = tensor_to_pil(img_tensor)
    masked_pil = tensor_to_pil(masked_img_tensor)

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_pil)
plt.title(f"Original Image\nPrediction: {prediction},\nProbability: {probs:.2f}")
plt.axis('off')

mask_pil = transforms.functional.to_pil_image(mask_parameter[0])
plt.subplot(1, 3, 2)
plt.imshow(mask_pil)
plt.title("Applied Perturbation Mask")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(masked_pil)
plt.title(f"Masked Image\nPrediction: {mask_prediction},\nProbability: {mask_probs:.2f}")
plt.axis('off')

plt.show()

# --- L2 Distance Report ------------------------------------------------------

l2_norm = torch.norm(img_tensor - masked_img_tensor, p=2)
print("Distance (L2 norm) between original image and masked image:\n---------------\n", l2_norm.item())

# --- Save Adversarial Image --------------------------------------------------

masked_img_tensor = img_tensor + mask_parameter

adv_pil = transforms.functional.to_pil_image(
    unnormalize(masked_img_tensor).squeeze(0).clamp(0, 1)
)
canvas = transforms.Resize(256)(img).copy()
canvas.paste(adv_pil, (16, 16))
canvas = canvas.resize(img.size, Image.LANCZOS)

canvas.save(fp='evasion_image.png')

new_img = Image.open("evasion_image.png")
with torch.no_grad():
    png_img_array = preprocess(new_img).to(device).unsqueeze(0)
    outputs = model(png_img_array)[0].argmax()

print("Target index is:", outputs)
print("Target label is:", labels[outputs])

# --- Verification Checks -----------------------------------------------------

# Check 1: classification
try:
    assessment_img = Image.open("evasion_image.png")
    idx = model(preprocess(assessment_img).unsqueeze(0).to(device))[0].argmax()
    print(idx, labels[idx])
except Exception as e:
    print(f"An error occurred: {e}")

# Check 2: image dimensions (should be 256x256)
try:
    print(assessment_img.size)
except Exception as e:
    print(f"An error occurred: {e}")

# Check 3: visual inspection (should still look like a rooster)
try:
    plt.imshow(assessment_img)
    plt.show()
except Exception as e:
    print(f"An error occurred: {e}")
