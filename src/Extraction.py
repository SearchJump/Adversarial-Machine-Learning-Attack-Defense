import sys
import os
import json
from collections import Counter
from matplotlib import pyplot as plt
from PIL import Image

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from datasets import load_dataset
from sklearn.metrics import roc_auc_score

# Try to upgrade dependencies in the DLI environment if available
try:
    import dependency_manager
    dependency_manager.upgrade_dependencies()
except ModuleNotFoundError:
    print("dependency_manager not found. Continuing with the current environment.")

# Set device configuration (will use GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ==========================================
# 1. Models and Setup Configuration
# ==========================================

# Load the victim ResNet-50 model
victim_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
victim_model.eval()

# Load ImageNet class labels
labels_file_path = "../data/labels.txt"
if os.path.exists(labels_file_path):
    with open(labels_file_path, 'r') as f:
        labels = [label.strip() for label in f.readlines()]
else:
    print(f"Warning: Labels file not found at {labels_file_path}.")

# Standard preprocessing for ResNet-50
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

# Define constants corresponding to indexes in food101 and ResNet-50
f101_pizza = 76
f101_hotdog = 55

rn_pizza = 963
rn_hotdog = 934

victim_label_list = [rn_hotdog, rn_pizza]

# ==========================================
# 2. Data Loading and Preprocessing
# ==========================================

def ds_preprocess(example):
    example['image'] = preprocess(example['image'])
    return example

print("Loading and filtering the food101 dataset...")
ds = load_dataset('ethz/food101')\
    .filter(lambda x: x['label'] in [f101_pizza, f101_hotdog])\
    .map(ds_preprocess).with_format('torch')

# ==========================================
# 3. Label Extraction Function
# ==========================================

def extract_labels(my_dataloader, my_victim_model):
    my_victim_model.eval()
    extracted_labels_result = []
    index = 0
    with torch.no_grad():
        for batch in my_dataloader:
            out = my_victim_model(batch['image'].to(device))
            proxy_scores = torch.nn.functional.softmax(out[:, [rn_hotdog, rn_pizza]], dim=1).cpu().numpy().argmax(1)
            for item in proxy_scores:
                extracted_labels_result.append([index, 'hotdog' if item == 0 else 'pizza'])
                index += 1
            print(f"Processed: {index}", end='\r')
    print()
    return extracted_labels_result

# Initialize the training dataloader without shuffling for extraction
dl = DataLoader(ds['train'], batch_size=32, shuffle=False)
print("Extracting labels from the victim model...")
extracted_labels_result = extract_labels(dl, victim_model)

# Relabel dataset with extracted labels
new_labels = [0 if label == 'hotdog' else 1 for idx, label in extracted_labels_result]

relabeled_train_ds = ds['train']\
    .remove_columns('label')\
    .add_column('label', new_labels)\
    .with_format('torch')

# ==========================================
# 4. Proxy Model (Copy-cat) Training
# ==========================================

proxy_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).to(device)
optimizer = torch.optim.Adam(proxy_model.parameters())

try:
    train_dl = DataLoader(relabeled_train_ds, batch_size=32, shuffle=True)
except Exception as e:
    print(f"Unable to initialize train_dl with relabeled_train_ds: {e}")
    sys.exit(1)

NUM_EPOCHS = 5
print(f"Training the proxy model for {NUM_EPOCHS} epochs...")
proxy_model.train()
for epoch in range(NUM_EPOCHS):
    for batch in train_dl:
        yhat = proxy_model(batch['image'].to(device))[:, [rn_hotdog, rn_pizza]]
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(yhat, batch['label'].to(device))
        loss.backward()
        optimizer.step()
        sys.stdout.write(f"\rEpoch {epoch+1}/{NUM_EPOCHS} - Loss: {loss.item():.4f}")
    print()

# ==========================================
# 5. Evaluation and Validation
# ==========================================

eval_dl = DataLoader(ds['validation'], batch_size=32, shuffle=False)
original_proxy_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1).to(device)

results_proxy = []
original_results_proxy = []
results_victim = []

proxy_model.eval()
original_proxy_model.eval()

print("Evaluating the model on the validation dataset...")
with torch.no_grad():
    for batch in eval_dl:
        images = batch['image'].to(device)
        
        # Trained proxy model
        yhat_p = proxy_model(images)[:, [rn_hotdog, rn_pizza]].cpu().numpy()
        results_proxy.append(yhat_p)

        # Victim model
        yhat_v = victim_model(images)[:, [rn_hotdog, rn_pizza]].cpu().numpy()
        results_victim.append(yhat_v)

        # Untrained proxy model (baseline)
        yhat_op = original_proxy_model(images)[:, [rn_hotdog, rn_pizza]].cpu().numpy()
        original_results_proxy.append(yhat_op)

final_pizza_scores_proxy = list(np.vstack(results_proxy)[:, 1])
final_pizza_labels_victim = np.vstack(results_victim).argmax(1)
final_pizza_labels = [1 if x == f101_pizza else 0 for x in ds['validation']['label']]

proxy_scores_dict = {'proxy_scores': final_pizza_scores_proxy}
original_proxy_scores = list(np.vstack(original_results_proxy)[:, 1])

# Calculate Metrics
counts_dict = Counter([x[1] for x in extracted_labels_result])
auc_before = roc_auc_score(final_pizza_labels_victim, original_proxy_scores)
auc_after_victim = roc_auc_score(final_pizza_labels_victim, final_pizza_scores_proxy)
auc_after_real = roc_auc_score(final_pizza_labels, final_pizza_scores_proxy)

print(f"""
Extraction Results:
    Extracted label counts: {dict(counts_dict)}
    
    AUC before training (with victim_model labels): {auc_before:.4f}
    AUC after training (with victim_model labels): {auc_after_victim:.4f}
    AUC after training (with original food101 labels): {auc_after_real:.4f}
""")

# ==========================================
# 6. Save Outputs
# ==========================================

os.makedirs("my_assessment", exist_ok=True)
try:
    with open("my_assessment/extraction_counts.json", 'w') as f:
        json.dump(counts_dict, f)
    np.save("my_assessment/extraction_proxy_scores.npy", proxy_scores_dict['proxy_scores'], allow_pickle=False)
    print("Results saved successfully in 'my_assessment/' directory.")
except Exception as e:
    print(f"An error occurred while saving: {e}")

# Restore previous environment dependencies if applicable
try:
    dependency_manager.restore_dependencies()
except NameError:
    pass
