import os
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import GaussianBlur
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_evasion_defense(image_path="evasion_image.png"):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found. Please run src/Evasion.py first to generate the adversarial image.")
        return

    # Load Model and Labels
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
    model.eval()

    labels_file = "../data/labels.txt" if os.path.exists("../data/labels.txt") else "data/labels.txt"
    if not os.path.exists(labels_file):
        labels_file = "labels.txt" # fallback
        
    with open(labels_file, 'r') as f:
        labels = [label.strip() for label in f.readlines()]

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load Adversarial Image
    img = Image.open(image_path)
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # Prediction 1: Without Defense
    with torch.no_grad():
        out_raw = model(img_tensor)
        idx_raw = out_raw[0].argmax().item()
        label_raw = labels[idx_raw]

    # Prediction 2: With Defense (Spatial Denoising / Gaussian Blur)
    # Applying a subtle blur disrupts high-frequency mathematical adjustments made by the optimizer
    denoiser = GaussianBlur(kernel_size=3, sigma=0.5)
    defended_tensor = denoiser(img_tensor)

    with torch.no_grad():
        out_defended = model(defended_tensor)
        idx_defended = out_defended[0].argmax().item()
        label_defended = labels[idx_defended]

    print("\n==================================================")
    print("           EVASION ATTACK DEFENSE EVALUATION      ")
    print("==================================================")
    print(f"Input Image Source:    {image_path}")
    print(f"Prediction WITHOUT Defense:  {label_raw} (Index {idx_raw})")
    print(f"Prediction WITH Defense:     {label_defended} (Index {idx_defended})")
    print("==================================================\n")

if __name__ == "__main__":
    run_evasion_defense()
