"""
OncoAI Fusion - Inference Service
Handles model loading and cancer classification predictions
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import io


class ResNet50MultiCancer(nn.Module):
    """ResNet50 with custom classification head for multi-cancer classification."""
    
    def __init__(self, num_classes=22, pretrained=False):
        super(ResNet50MultiCancer, self).__init__()
        
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


class CancerClassifier:
    """Service for cancer classification inference."""
    
    # Cancer class mapping with detailed information
    CLASS_INFO = {
        'brain_glioma': {'name': 'Brain Glioma', 'category': 'Brain Cancer', 'severity': 'High'},
        'brain_menin': {'name': 'Brain Meningioma', 'category': 'Brain Cancer', 'severity': 'Medium'},
        'brain_tumor': {'name': 'Brain Tumor', 'category': 'Brain Cancer', 'severity': 'High'},
        'breast_benign': {'name': 'Breast Benign', 'category': 'Breast Cancer', 'severity': 'Low'},
        'breast_malignant': {'name': 'Breast Malignant', 'category': 'Breast Cancer', 'severity': 'High'},
        'cervix_dyk': {'name': 'Cervix Dyskeratotic', 'category': 'Cervical Cancer', 'severity': 'Medium'},
        'cervix_koc': {'name': 'Cervix Koilocytotic', 'category': 'Cervical Cancer', 'severity': 'Medium'},
        'cervix_mep': {'name': 'Cervix Metaplastic', 'category': 'Cervical Cancer', 'severity': 'Low'},
        'cervix_pab': {'name': 'Cervix Parabasal', 'category': 'Cervical Cancer', 'severity': 'Low'},
        'cervix_sfi': {'name': 'Cervix Superficial-Intermediate', 'category': 'Cervical Cancer', 'severity': 'Low'},
        'colon_aca': {'name': 'Colon Adenocarcinoma', 'category': 'Colon Cancer', 'severity': 'High'},
        'colon_bnt': {'name': 'Colon Benign Tissue', 'category': 'Colon Cancer', 'severity': 'Low'},
        'kidney_normal': {'name': 'Kidney Normal', 'category': 'Kidney Cancer', 'severity': 'None'},
        'kidney_tumor': {'name': 'Kidney Tumor', 'category': 'Kidney Cancer', 'severity': 'High'},
        'lung_aca': {'name': 'Lung Adenocarcinoma', 'category': 'Lung Cancer', 'severity': 'High'},
        'lung_bnt': {'name': 'Lung Benign Tissue', 'category': 'Lung Cancer', 'severity': 'Low'},
        'lung_scc': {'name': 'Lung Squamous Cell Carcinoma', 'category': 'Lung Cancer', 'severity': 'High'},
        'lymph_cll': {'name': 'Chronic Lymphocytic Leukemia', 'category': 'Lymphoma', 'severity': 'Medium'},
        'lymph_fl': {'name': 'Follicular Lymphoma', 'category': 'Lymphoma', 'severity': 'Medium'},
        'lymph_mcl': {'name': 'Mantle Cell Lymphoma', 'category': 'Lymphoma', 'severity': 'High'},
        'oral_normal': {'name': 'Oral Normal', 'category': 'Oral Cancer', 'severity': 'None'},
        'oral_scc': {'name': 'Oral Squamous Cell Carcinoma', 'category': 'Oral Cancer', 'severity': 'High'},
    }
    
    CLASS_NAMES = list(CLASS_INFO.keys())
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the classifier with optional model path."""
        self.device = self._get_device()
        self.model = None
        self.transform = self._get_transform()
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def _get_device(self) -> str:
        """Get the best available device."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _get_transform(self) -> transforms.Compose:
        """Get inference transforms."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_model(self, model_path: str) -> bool:
        """Load model from checkpoint."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            num_classes = checkpoint.get('config', {}).get('num_classes', 22)
            self.model = ResNet50MultiCancer(num_classes=num_classes, pretrained=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded from {model_path}")
            print(f"   Validation accuracy: {checkpoint.get('val_acc', 'N/A')}%")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for inference."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        tensor = self.transform(image)
        return tensor.unsqueeze(0)  # Add batch dimension
    
    def predict(self, image: Image.Image) -> Dict:
        """
        Predict cancer type from image.
        
        Returns:
            Dict with prediction results including class, confidence, and metadata
        """
        if self.model is None:
            return {
                'status': 'error',
                'message': 'Model not loaded. Please train or load a model first.'
            }
        
        try:
            # Preprocess
            input_tensor = self.preprocess_image(image).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Get class info
            predicted_class = self.CLASS_NAMES[predicted_idx.item()]
            class_info = self.CLASS_INFO[predicted_class]
            
            # Get top-5 predictions
            top5_probs, top5_indices = torch.topk(probabilities, 5)
            top5_predictions = [
                {
                    'class': self.CLASS_NAMES[idx.item()],
                    'name': self.CLASS_INFO[self.CLASS_NAMES[idx.item()]]['name'],
                    'confidence': prob.item()
                }
                for prob, idx in zip(top5_probs[0], top5_indices[0])
            ]
            
            return {
                'status': 'success',
                'prediction': {
                    'class': predicted_class,
                    'name': class_info['name'],
                    'category': class_info['category'],
                    'severity': class_info['severity'],
                    'confidence': confidence.item()
                },
                'top5_predictions': top5_predictions,
                'model_info': {
                    'device': self.device,
                    'num_classes': len(self.CLASS_NAMES)
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def predict_from_bytes(self, image_bytes: bytes) -> Dict:
        """Predict from image bytes."""
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return self.predict(image)
    
    def predict_from_path(self, image_path: str) -> Dict:
        """Predict from image file path."""
        image = Image.open(image_path).convert('RGB')
        return self.predict(image)


# Global classifier instance
_classifier: Optional[CancerClassifier] = None


def get_classifier() -> CancerClassifier:
    """Get or create the global classifier instance."""
    global _classifier
    if _classifier is None:
        # Try to load best model if available
        model_path = Path(__file__).parent.parent.parent / 'ml_models' / 'checkpoints' / 'best_model.pth'
        _classifier = CancerClassifier(str(model_path) if model_path.exists() else None)
    return _classifier


def initialize_classifier(model_path: str) -> CancerClassifier:
    """Initialize classifier with specific model path."""
    global _classifier
    _classifier = CancerClassifier(model_path)
    return _classifier
