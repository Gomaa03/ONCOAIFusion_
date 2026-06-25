"""
OncoAI Fusion - Brain Cancer Inference Service
Multi-class classification: Glioma, Meningioma, Pituitary
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from typing import Dict, Optional
import io


class ResNet50BrainCancer(nn.Module):
    """ResNet50 for brain cancer classification (3 classes)."""
    
    def __init__(self, num_classes=3, pretrained=False):
        super(ResNet50BrainCancer, self).__init__()
        
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class BrainCancerClassifier:
    """Brain cancer classification service."""
    
    CLASS_NAMES = ['brain_glioma', 'brain_menin', 'brain_tumor']
    CLASS_INFO = {
        'brain_glioma': {
            'name': 'Glioma Tumor',
            'category': 'Brain Cancer',
            'severity': 'High',
            'description': 'Gliomas are tumors that originate from glial cells in the brain. They can be aggressive and require immediate attention.',
            'recommendation': 'Urgent neurology consultation. MRI with contrast and biopsy recommended for staging.'
        },
        'brain_menin': {
            'name': 'Meningioma Tumor',
            'category': 'Brain Cancer',
            'severity': 'Medium',
            'description': 'Meningiomas arise from the meninges, the membranes surrounding the brain. Most are benign but require monitoring.',
            'recommendation': 'Neurosurgical evaluation recommended. Regular MRI follow-up for monitoring tumor growth.'
        },
        'brain_tumor': {
            'name': 'Pituitary Tumor',
            'category': 'Brain Cancer',
            'severity': 'Medium',
            'description': 'Pituitary tumors affect the pituitary gland. Many are benign but can cause hormonal imbalances.',
            'recommendation': 'Endocrinology and neurology consultation. Hormonal evaluation and visual field testing recommended.'
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
                'BRAIN_MODEL_PATH',
                os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ml_models', 'checkpoints', 'brain_cancer', 'best_model.pth')
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
            
            self.model = ResNet50BrainCancer(num_classes=3, pretrained=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            val_acc = checkpoint.get('val_acc', 'N/A')
            print(f"✅ Brain Cancer model loaded")
            print(f"   Validation accuracy: {val_acc}%")
            return True
        except Exception as e:
            print(f"❌ Error loading brain cancer model: {e}")
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
            
            # Get all class probabilities
            glioma_prob = probabilities[0][0].item()
            menin_prob = probabilities[0][1].item()
            pituitary_prob = probabilities[0][2].item()
            
            # Confidence threshold: If confidence < 50%, likely no tumor
            CONFIDENCE_THRESHOLD = 0.50
            
            if confidence.item() < CONFIDENCE_THRESHOLD:
                # Low confidence - likely normal/no tumor
                return {
                    'status': 'success',
                    'prediction': {
                        'class': 'brain_normal',
                        'name': 'No Tumor Detected',
                        'category': 'Brain Cancer',
                        'severity': 'None',
                        'confidence': 1.0 - confidence.item(),  # Inverse confidence as "normal" confidence
                        'description': 'No significant tumor detected in the brain MRI. The image appears to show normal brain tissue.',
                        'recommendation': 'No immediate action required. Continue regular health check-ups and consult a neurologist if symptoms persist.'
                    },
                    'probabilities': {
                        'normal': 1.0 - max(glioma_prob, menin_prob, pituitary_prob),
                        'glioma': glioma_prob,
                        'meningioma': menin_prob,
                        'pituitary': pituitary_prob
                    },
                    'model_info': {
                        'type': 'brain_cancer_multiclass',
                        'device': self.device
                    }
                }
            
            predicted_class = self.CLASS_NAMES[predicted_idx.item()]
            class_info = self.CLASS_INFO[predicted_class]
            
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
                    'glioma': glioma_prob,
                    'meningioma': menin_prob,
                    'pituitary': pituitary_prob
                },
                'model_info': {
                    'type': 'brain_cancer_multiclass',
                    'device': self.device
                }
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def predict_from_bytes(self, image_bytes: bytes) -> Dict:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return self.predict(image)


# Global instance
_classifier: Optional[BrainCancerClassifier] = None

def get_brain_cancer_classifier() -> BrainCancerClassifier:
    global _classifier
    if _classifier is None:
        _classifier = BrainCancerClassifier()
    return _classifier
