import numpy as np
from sklearn.ensemble import IsolationForest
from src.Poisoning import get_data

def run_poisoning_defense():
    print("Loading poisoned dataset...")
    data = get_data()
    x_train = data["x_train"]
    y_train = data["y_train"]

    # Flatten image tensors for basic statistical pixel distribution anomaly detection
    flat_features = x_train.reshape(x_train.shape[0], -1)

    # Target class 6: Frogs (which have been partially overwritten with Cat images)
    target_class_idx = 6
    target_indices = np.where(y_train.argmax(axis=1) == target_class_idx)[0]
    target_features = flat_features[target_indices]

    print(f"Analyzing {len(target_indices)} samples inside Frog class (Index 6) for anomalies...")

    # We use an Isolation Forest with contamination matching the suspected poison density (~10-15%)
    clf = IsolationForest(contamination=0.15, random_state=42, n_jobs=-1)
    predictions = clf.fit_predict(target_features) # -1 indicates anomaly/poisoned

    poisoned_detected = target_indices[predictions == -1]
    clean_detected = target_indices[predictions == 1]

    print("\n==================================================")
    print("          POISONING DEFENSE SANITIZATION AUDIT    ")
    print("==================================================")
    print(f"Total Frog Class Samples Analyzed:    {len(target_indices)}")
    print(f"Anomalous (Poisoned) Samples Pruned:  {len(poisoned_detected)}")
    print(f"Clean (Legitimate) Samples Preserved: {len(clean_detected)}")
    print("==================================================\n")

if __name__ == "__main__":
    run_poisoning_defense()
