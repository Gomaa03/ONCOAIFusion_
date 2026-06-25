"""
OncoAI Fusion - Cervical Cancer Inference Service
Multi-class classification: DYK, KOC, MEP, PAB, SFI (5 cell types)
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from typing import Dict, Optional
import io


class ResNet50CervicalCancer(nn.Module):
    """ResNet50 for cervical cancer classification (5 classes)."""
    
    def __init__(self, num_classes=5, pretrained=False):
        super(ResNet50CervicalCancer, self).__init__()
        
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


class CervicalCancerClassifier:
    """Cervical cancer classification service."""
    
    CLASS_NAMES = ['cervix_dyk', 'cervix_koc', 'cervix_mep', 'cervix_pab', 'cervix_sfi']
    CLASS_INFO = {
        'cervix_dyk': {
            'name': 'Dyskeratotic Cells',
            'category': 'Cervical Cancer',
            'severity': 'High',
            'description': 'Dyskeratotic cells show abnormal keratinization, often associated with HPV infection and precancerous changes.',
            'recommendation': 'Colposcopy and biopsy recommended. HPV testing advised.'
        },
        'cervix_koc': {
            'name': 'Koilocytotic Cells',
            'category': 'Cervical Cancer',
            'severity': 'Medium',
            'description': 'Koilocytes are cells with perinuclear clearing, a hallmark of HPV infection.',
            'recommendation': 'HPV typing and follow-up Pap smear recommended. Colposcopy may be indicated.'
        },
        'cervix_mep': {
            'name': 'Metaplastic Cells',
            'category': 'Cervical Cancer',
            'severity': 'Low',
            'description': 'Metaplastic cells are normal findings in the transformation zone. Usually benign.',
            'recommendation': 'Routine follow-up. No immediate intervention required.'
        },
        'cervix_pab': {
            'name': 'Parabasal Cells',
            'category': 'Cervical Cancer',
            'severity': 'Low',
            'description': 'Parabasal cells are immature squamous cells, often seen in atrophic conditions.',
            'recommendation': 'Evaluate hormonal status if symptomatic. Routine monitoring sufficient.'
        },
        'cervix_sfi': {
            'name': 'Superficial-Intermediate Cells',
            'category': 'Cervical Cancer',
            'severity': 'None',
            'description': 'Normal mature squamous cells from the cervical epithelium. No abnormality detected.',
            'recommendation': 'Normal finding. Continue routine screening as per guidelines.'
        }
    }
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = self._get_device()
        self.model = None
        self.transform = self._get_transform()
        
        if model_path is None:
            import os
            # Use environment variable or default path (works in Docker and local)
            model_path = os.environ.get(
                'CERVICAL_MODEL_PATH',
                os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ml_models', 'checkpoints', 'cervical_cancer', 'best_model.pth')
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
            
            self.model = ResNet50CervicalCancer(num_classes=5, pretrained=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            val_acc = checkpoint.get('val_acc', 'N/A')
            print(f"✅ Cervical Cancer model loaded")
            print(f"   Validation accuracy: {val_acc}%")
            return True
        except Exception as e:
            print(f"❌ Error loading cervical cancer model: {e}")
            return False
    
    def predict(self, image: Image.Image) -> Dict:
        if self.model is None:
            return {
                'status': 'error',
                'message': 'Model not loaded'
            }
        
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
            
            # Get all class probabilities
            dyk_prob = probabilities[0][0].item()
            koc_prob = probabilities[0][1].item()
            mep_prob = probabilities[0][2].item()
            pab_prob = probabilities[0][3].item()
            sfi_prob = probabilities[0][4].item()
            
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
                    'dyskeratotic': dyk_prob,
                    'koilocytotic': koc_prob,
                    'metaplastic': mep_prob,
                    'parabasal': pab_prob,
                    'superficial_intermediate': sfi_prob
                },
                'model_info': {
                    'type': 'cervical_cancer_multiclass',
                    'device': self.device
                }
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def predict_from_bytes(self, image_bytes: bytes) -> Dict:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return self.predict(image)


# Global instance
_classifier: Optional[CervicalCancerClassifier] = None

def get_cervical_cancer_classifier() -> CervicalCancerClassifier:
    global _classifier
    if _classifier is None:
        _classifier = CervicalCancerClassifier()
    return _classifier
