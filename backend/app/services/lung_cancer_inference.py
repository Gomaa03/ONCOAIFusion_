"""
OncoAI Fusion - Lung Cancer Inference Service
Multi-class classification: Adenocarcinoma, Benign, Squamous Cell
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from typing import Dict, Optional
import io


class ResNet50LungCancer(nn.Module):
    """ResNet50 for lung cancer classification (3 classes)."""
    
    def __init__(self, num_classes=3, pretrained=False):
        super(ResNet50LungCancer, self).__init__()
        
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class LungCancerClassifier:
    """Lung cancer classification service."""
    
    CLASS_NAMES = ['lung_aca', 'lung_bnt', 'lung_scc']
    CLASS_INFO = {
        'lung_aca': {
            'name': 'Lung Adenocarcinoma',
            'category': 'Lung Cancer',
            'severity': 'High',
            'description': 'Adenocarcinoma detected - the most common type of lung cancer, typically arising in peripheral lung tissue.',
            'recommendation': 'Urgent oncology referral required. Additional staging with PET-CT and tissue biopsy recommended.'
        },
        'lung_bnt': {
            'name': 'Benign Lung Tissue',
            'category': 'Lung Cancer',
            'severity': 'None',
            'description': 'Normal benign lung tissue with no detectable malignancy.',
            'recommendation': 'No immediate action required. Continue routine health monitoring.'
        },
        'lung_scc': {
            'name': 'Squamous Cell Carcinoma',
            'category': 'Lung Cancer',
            'severity': 'High',
            'description': 'Squamous cell carcinoma detected - often associated with smoking, typically arising in central airways.',
            'recommendation': 'Urgent oncology referral required. Bronchoscopy and staging workup recommended.'
        }
    }
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = self._get_device()
        self.model = None
        self.transform = self._get_transform()
        
        if model_path is None:
            import os
            model_path = os.environ.get(
                'LUNG_MODEL_PATH',
                os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ml_models', 'checkpoints', 'lung_cancer', 'best_model.pth')
            )
        
        if Path(model_path).exists():
            self.load_model(str(model_path))
    
    def _get_device(self) -> str:
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path: str) -> bool:
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            self.model = ResNet50LungCancer(num_classes=3, pretrained=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            val_acc = checkpoint.get('val_acc', 'N/A')
            print(f"✅ Lung Cancer model loaded")
            print(f"   Validation accuracy: {val_acc}%")
            return True
        except Exception as e:
            print(f"❌ Error loading lung cancer model: {e}")
            return False
    
    def predict(self, image: Image.Image) -> Dict:
        if self.model is None:
            return {'status': 'error', 'message': 'Model not loaded'}
        
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            predicted_class = self.CLASS_NAMES[predicted_idx.item()]
            class_info = self.CLASS_INFO[predicted_class]
            
            # Get class probabilities
            aca_prob = probabilities[0][0].item()
            bnt_prob = probabilities[0][1].item()
            scc_prob = probabilities[0][2].item()
            
            return {
                'status': 'success',
                'prediction': {
                    'class': predicted_class,
                    'name': class_info['name'],
                    'category': class_info['category'],
                    'severity': class_info['severity'],
                    'confidence': confidence.item(),
                    'description': class_info['description'],
                    'recommendation': class_info['recommendation']
                },
                'probabilities': {
                    'adenocarcinoma': aca_prob,
                    'benign': bnt_prob,
                    'squamous_cell': scc_prob
                },
                'model_info': {
                    'type': 'lung_cancer_multiclass',
                    'device': self.device
                }
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def predict_from_bytes(self, image_bytes: bytes) -> Dict:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return self.predict(image)


# Global instance
_classifier: Optional[LungCancerClassifier] = None

def get_lung_cancer_classifier() -> LungCancerClassifier:
    global _classifier
    if _classifier is None:
        _classifier = LungCancerClassifier()
    return _classifier
