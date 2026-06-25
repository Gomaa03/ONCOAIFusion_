"""
OncoAI Fusion - Display Model Accuracies
Run: python3 check_accuracies.py
"""

import torch
import os

print("\n" + "=" * 50)
print("📊 OncoAI Fusion - Trained Model Accuracies")
print("=" * 50 + "\n")

base_path = os.path.dirname(os.path.abspath(__file__))
checkpoints_path = os.path.join(base_path, "..", "checkpoints")

models = {
    "Brain Cancer": "brain_cancer",
    "Breast Cancer": "breast_cancer", 
    "Cervical Cancer": "cervical_cancer",
    "Kidney Cancer": "kidney_cancer",
    "Lung Cancer": "lung_cancer",
    "Colon Cancer": "colon_cancer",
    "Lymphoma": "lymphoma",
    "Oral Cancer": "oral_cancer"
}

for name, folder in models.items():
    model_path = os.path.join(checkpoints_path, folder, "best_model.pth")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        accuracy = checkpoint.get("val_acc", "N/A")
        if isinstance(accuracy, (int, float)):
            print(f"  {name:20} → {accuracy:.2f}%")
        else:
            print(f"  {name:20} → {accuracy}")
    else:
        print(f"  {name:20} → Not trained yet")

print("\n" + "=" * 50 + "\n")
