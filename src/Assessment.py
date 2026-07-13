import os
import json
import torch
import spacy
from transformers import AutoTokenizer, RobertaForSequenceClassification
from alibi.utils import spacy_model
from alibi.explainers import AnchorText

# Set device (defaulting to CUDA if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ==========================================
# 1. Define the Detector
# ==========================================

class FakeDetector:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")
        self.model = RobertaForSequenceClassification.from_pretrained("openai-community/roberta-base-openai-detector")
        self.model.to(device)

    def is_robot(self, prompt: str) -> bool:
        with torch.no_grad():
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
            logits = self.model(**inputs).logits
            pred = self.model.config.id2label[logits.argmax().item()]
            return pred == "Fake"

# Instantiate and verify the detector
print("Instantiating Detector...")
f = FakeDetector()

test_string = (
    "The capital of France is Paris. It is the largest city in France and serves as the country's "
    "main cultural and commercial center. Paris has a rich history and is known for its significant "
    "influence in areas such as finance, diplomacy, commerce, culture, fashion, and gastronomy. "
    "With an estimated population of over 2 million residents, Paris is a vibrant and modern city "
    "with a unique tourist appeal. "
)

print(f"The string is fake: {f.is_robot(test_string)}")

# ==========================================
# 2. Set Up the Prediction Model for Alibi
# ==========================================

model_nlp_name = 'en_core_web_md'
spacy_model(model=model_nlp_name)
nlp = spacy.load(model_nlp_name)

tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")
model = RobertaForSequenceClassification.from_pretrained("openai-community/roberta-base-openai-detector")
model.to(device)

def predict(x):
    inputs = tokenizer(x, return_tensors="pt", padding=True)
    with torch.no_grad():
        output = model(**(inputs).to(device))
    return output.logits.cpu().numpy()

# ==========================================
# 3. Instantiate Explainer and Run Explanation
# ==========================================

print("Instantiating AnchorText explainer...")
explainer = AnchorText(
    predictor=predict,
    sampling_strategy='similarity',
    nlp=nlp,
    use_proba=False
)

print("Running explanation on the test string...")
explanation = explainer.explain(
    test_string,
    threshold=0.95
)

# Output results to console
print(f"Anchor: {explanation.anchor}")
print(f"Precision: {explanation.precision:.2f}\n")

# ==========================================
# 4. Save the Submission File
# ==========================================

os.makedirs("my_assessment", exist_ok=True)
try:
    # Save the three words that most contributed to the "Fake" classification
    submission_words = explanation.anchor[:3]
    with open("my_assessment/assessments_submission.json", 'w') as f_out:
        json.dump(submission_words, f_out)
    print("Successfully saved top 3 anchors to 'my_assessment/assessments_submission.json'.")
except Exception as e:
    print(f"An error occurred during save operations: {e}")
