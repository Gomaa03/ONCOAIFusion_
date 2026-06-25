"""
OncoAI Fusion - Unified Prediction API Routes
Auto-detects cancer type (Brain/Breast/Cervical/Kidney/Lung/Colon/Lymphoma/Oral) from uploaded image
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image
import io

router = APIRouter()

# Try to import inference services
try:
    from app.services.breast_cancer_inference import get_breast_cancer_classifier
    BREAST_INFERENCE_AVAILABLE = True
except ImportError:
    BREAST_INFERENCE_AVAILABLE = False

try:
    from app.services.brain_cancer_inference import get_brain_cancer_classifier
    BRAIN_INFERENCE_AVAILABLE = True
except ImportError:
    BRAIN_INFERENCE_AVAILABLE = False

try:
    from app.services.cervical_cancer_inference import get_cervical_cancer_classifier
    CERVICAL_INFERENCE_AVAILABLE = True
except ImportError:
    CERVICAL_INFERENCE_AVAILABLE = False

try:
    from app.services.kidney_cancer_inference import get_kidney_cancer_classifier
    KIDNEY_INFERENCE_AVAILABLE = True
except ImportError:
    KIDNEY_INFERENCE_AVAILABLE = False

try:
    from app.services.lung_cancer_inference import get_lung_cancer_classifier
    LUNG_INFERENCE_AVAILABLE = True
except ImportError:
    LUNG_INFERENCE_AVAILABLE = False

try:
    from app.services.colon_cancer_inference import get_colon_cancer_classifier
    COLON_INFERENCE_AVAILABLE = True
except ImportError:
    COLON_INFERENCE_AVAILABLE = False

try:
    from app.services.lymphoma_inference import get_lymphoma_classifier
    LYMPHOMA_INFERENCE_AVAILABLE = True
except ImportError:
    LYMPHOMA_INFERENCE_AVAILABLE = False

try:
    from app.services.oral_cancer_inference import get_oral_cancer_classifier
    ORAL_INFERENCE_AVAILABLE = True
except ImportError:
    ORAL_INFERENCE_AVAILABLE = False


def detect_image_type(image: Image.Image) -> str:
    """
    Detect if image is brain MRI, breast histopathology, or cervical cytology.
    Brain MRI: Usually grayscale, darker, shows brain structure
    Breast histopathology: Usually colorful (pink/purple stained tissue, H&E stain)
    Cervical cytology: Green-dominant (Papanicolaou staining), moderate saturation
    """
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Sample pixels to analyze color distribution
    img_small = image.resize((100, 100))
    pixels = list(img_small.getdata())
    
    # Calculate color statistics
    total_pixels = len(pixels)
    grayscale_count = 0
    pink_purple_count = 0
    green_dominant_count = 0
    light_background_count = 0
    
    for r, g, b in pixels:
        # Check if pixel is roughly grayscale (R ≈ G ≈ B)
        max_diff = max(abs(r-g), abs(g-b), abs(r-b))
        if max_diff < 30:
            grayscale_count += 1
        
        # Check for pink/purple tones (common in breast H&E staining)
        if r > 150 and b > 100 and g < r and g < b:
            pink_purple_count += 1
        
        # Check for green-dominant tones (common in cervical Papanicolaou staining)
        if g > r and g > b:
            green_dominant_count += 1
        
        # Check for light background
        if r > 200 and g > 200 and b > 200:
            light_background_count += 1
    
    grayscale_ratio = grayscale_count / total_pixels
    pink_ratio = pink_purple_count / total_pixels
    green_ratio = green_dominant_count / total_pixels
    light_bg_ratio = light_background_count / total_pixels
    
    # Brain MRI images are predominantly grayscale
    if grayscale_ratio > 0.7:
        return 'brain'
    
    # Cervical cytology: green-dominant (Papanicolaou staining produces greenish cells)
    # Typically has high green ratio (>0.4) and moderate grayscale
    if green_ratio > 0.4 and grayscale_ratio > 0.3:
        return 'cervical'
    
    # Breast histopathology has pink/purple staining (H&E stain)
    if pink_ratio > 0.1:
        return 'breast'
    
    # Default heuristic: check average saturation and hue
    import colorsys
    saturations = []
    hues = []
    for r, g, b in pixels[:1000]:
        h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
        saturations.append(s)
        hues.append(h)
    avg_saturation = sum(saturations) / len(saturations)
    avg_hue = sum(hues) / len(hues)
    
    # Low saturation = grayscale = brain MRI
    if avg_saturation < 0.15:
        return 'brain'
    # Green hue range (0.15-0.40) with moderate saturation = cervical
    elif 0.15 < avg_hue < 0.40 and avg_saturation < 0.4:
        return 'cervical'
    else:
        return 'breast'


@router.post('/predict')
async def predict(file: UploadFile = File(...), cancer_type: str = None):
    """
    Unified cancer prediction endpoint.
    
    Args:
        file: Image file to analyze
        cancer_type: Optional. Explicitly specify 'brain', 'breast', 'cervical', 'kidney', 'lung', 'colon', 'lymphoma', or 'oral'.
                     If not specified, auto-detection is used (less reliable).
    """
    try:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Use explicit type if provided, otherwise auto-detect
        if cancer_type and cancer_type.lower() in ['brain', 'breast', 'cervical', 'kidney', 'lung', 'colon', 'lymphoma', 'oral']:
            detected_type = cancer_type.lower()
        else:
            detected_type = detect_image_type(image)
        
        if detected_type == 'brain':
            # Brain cancer classification
            if BRAIN_INFERENCE_AVAILABLE:
                classifier = get_brain_cancer_classifier()
                if classifier.model is not None:
                    result = classifier.predict(image)
                    result['detected_image_type'] = 'brain_mri'
                    return result
            
            # Demo mode for brain
            import random
            demo_types = [
                {'class': 'brain_glioma', 'name': 'Glioma Tumor', 'severity': 'High',
                 'description': 'Gliomas originate from glial cells in the brain.',
                 'recommendation': 'Urgent neurology consultation recommended.'},
                {'class': 'brain_menin', 'name': 'Meningioma Tumor', 'severity': 'Medium',
                 'description': 'Meningiomas arise from the meninges surrounding the brain.',
                 'recommendation': 'Neurosurgical evaluation recommended.'},
                {'class': 'brain_tumor', 'name': 'Pituitary Tumor', 'severity': 'Medium',
                 'description': 'Pituitary tumors affect the pituitary gland.',
                 'recommendation': 'Endocrinology consultation recommended.'}
            ]
            demo = random.choice(demo_types)
            return {
                'status': 'demo',
                'message': 'Brain model not loaded. Using demo response.',
                'detected_image_type': 'brain_mri',
                'prediction': {**demo, 'category': 'Brain Cancer', 'confidence': 0.87},
                'probabilities': {
                    'glioma': 0.87 if demo['class'] == 'brain_glioma' else 0.05,
                    'meningioma': 0.87 if demo['class'] == 'brain_menin' else 0.06,
                    'pituitary': 0.87 if demo['class'] == 'brain_tumor' else 0.07
                }
            }
        
        elif detected_type == 'cervical':
            # Cervical cancer classification
            if CERVICAL_INFERENCE_AVAILABLE:
                classifier = get_cervical_cancer_classifier()
                if classifier.model is not None:
                    result = classifier.predict(image)
                    result['detected_image_type'] = 'cervical_cytology'
                    return result
            
            # Demo mode for cervical
            import random
            demo_types = [
                {'class': 'cervix_dyk', 'name': 'Dyskeratotic Cells', 'severity': 'High',
                 'description': 'Abnormal keratinization associated with HPV infection.',
                 'recommendation': 'Colposcopy and biopsy recommended.'},
                {'class': 'cervix_koc', 'name': 'Koilocytotic Cells', 'severity': 'Medium',
                 'description': 'Cells with perinuclear clearing, hallmark of HPV.',
                 'recommendation': 'HPV typing and follow-up Pap smear recommended.'},
                {'class': 'cervix_sfi', 'name': 'Superficial-Intermediate Cells', 'severity': 'None',
                 'description': 'Normal mature squamous cells.',
                 'recommendation': 'Normal finding. Continue routine screening.'}
            ]
            demo = random.choice(demo_types)
            return {
                'status': 'demo',
                'message': 'Cervical model not loaded. Using demo response.',
                'detected_image_type': 'cervical_cytology',
                'prediction': {**demo, 'category': 'Cervical Cancer', 'confidence': 0.85},
                'probabilities': {
                    'dyskeratotic': 0.20,
                    'koilocytotic': 0.20,
                    'metaplastic': 0.20,
                    'parabasal': 0.20,
                    'superficial_intermediate': 0.20
                }
            }
        
        elif detected_type == 'kidney':
            # Kidney cancer classification
            if KIDNEY_INFERENCE_AVAILABLE:
                classifier = get_kidney_cancer_classifier()
                if classifier.model is not None:
                    result = classifier.predict(image)
                    result['detected_image_type'] = 'kidney_scan'
                    return result
            
            # Demo mode for kidney
            import random
            demo_types = [
                {'class': 'kidney_normal', 'name': 'Normal Kidney Tissue', 'severity': 'None',
                 'description': 'Normal kidney tissue with no detectable tumor.',
                 'recommendation': 'No immediate action required.'},
                {'class': 'kidney_tumor', 'name': 'Kidney Tumor', 'severity': 'High',
                 'description': 'Tumor detected in kidney tissue.',
                 'recommendation': 'Urgent referral to urologist/oncologist.'}
            ]
            demo = random.choice(demo_types)
            return {
                'status': 'demo',
                'message': 'Kidney model not loaded. Using demo response.',
                'detected_image_type': 'kidney_scan',
                'prediction': {**demo, 'category': 'Kidney Cancer', 'confidence': 0.88},
                'probabilities': {
                    'normal': 0.50,
                    'tumor': 0.50
                }
            }
        
        elif detected_type == 'lung':
            # Lung cancer classification
            if LUNG_INFERENCE_AVAILABLE:
                classifier = get_lung_cancer_classifier()
                if classifier.model is not None:
                    result = classifier.predict(image)
                    result['detected_image_type'] = 'lung_scan'
                    return result
            
            # Demo mode for lung
            import random
            demo_types = [
                {'class': 'lung_aca', 'name': 'Lung Adenocarcinoma', 'severity': 'High',
                 'description': 'Adenocarcinoma detected in lung tissue.',
                 'recommendation': 'Urgent oncology referral required.'},
                {'class': 'lung_bnt', 'name': 'Benign Lung Tissue', 'severity': 'None',
                 'description': 'Normal benign lung tissue.',
                 'recommendation': 'No immediate action required.'},
                {'class': 'lung_scc', 'name': 'Squamous Cell Carcinoma', 'severity': 'High',
                 'description': 'Squamous cell carcinoma detected.',
                 'recommendation': 'Urgent oncology referral required.'}
            ]
            demo = random.choice(demo_types)
            return {
                'status': 'demo',
                'message': 'Lung model not loaded. Using demo response.',
                'detected_image_type': 'lung_scan',
                'prediction': {**demo, 'category': 'Lung Cancer', 'confidence': 0.85},
                'probabilities': {
                    'adenocarcinoma': 0.33,
                    'benign': 0.34,
                    'squamous_cell': 0.33
                }
            }
        
        elif detected_type == 'colon':
            # Colon cancer classification
            if COLON_INFERENCE_AVAILABLE:
                classifier = get_colon_cancer_classifier()
                if classifier.model is not None:
                    result = classifier.predict(image)
                    result['detected_image_type'] = 'colon_histopathology'
                    return result
            
            # Demo mode for colon
            import random
            demo_types = [
                {'class': 'colon_aca', 'name': 'Colon Adenocarcinoma', 'severity': 'High',
                 'description': 'Adenocarcinoma detected in colon tissue.',
                 'recommendation': 'Urgent oncology referral required.'},
                {'class': 'colon_bnt', 'name': 'Benign Colon Tissue', 'severity': 'None',
                 'description': 'Normal benign colon tissue.',
                 'recommendation': 'No immediate action required.'}
            ]
            demo = random.choice(demo_types)
            return {
                'status': 'demo',
                'message': 'Colon model not loaded. Using demo response.',
                'detected_image_type': 'colon_histopathology',
                'prediction': {**demo, 'category': 'Colon Cancer', 'confidence': 0.89},
                'probabilities': {
                    'adenocarcinoma': 0.50,
                    'benign': 0.50
                }
            }
        
        elif detected_type == 'lymphoma':
            # Lymphoma classification
            if LYMPHOMA_INFERENCE_AVAILABLE:
                classifier = get_lymphoma_classifier()
                if classifier.model is not None:
                    result = classifier.predict(image)
                    result['detected_image_type'] = 'lymphoma_histopathology'
                    return result
            
            # Demo mode for lymphoma
            import random
            demo_types = [
                {'class': 'lymph_cll', 'name': 'Chronic Lymphocytic Leukemia', 'severity': 'High',
                 'description': 'CLL detected in blood cells.',
                 'recommendation': 'Urgent hematology/oncology referral required.'},
                {'class': 'lymph_fl', 'name': 'Follicular Lymphoma', 'severity': 'Medium',
                 'description': 'Follicular lymphoma detected.',
                 'recommendation': 'Oncology referral recommended.'},
                {'class': 'lymph_mcl', 'name': 'Mantle Cell Lymphoma', 'severity': 'High',
                 'description': 'MCL detected.',
                 'recommendation': 'Urgent oncology referral required.'}
            ]
            demo = random.choice(demo_types)
            return {
                'status': 'demo',
                'message': 'Lymphoma model not loaded. Using demo response.',
                'detected_image_type': 'lymphoma_histopathology',
                'prediction': {**demo, 'category': 'Lymphoma', 'confidence': 0.87},
                'probabilities': {
                    'cll': 0.33,
                    'fl': 0.34,
                    'mcl': 0.33
                }
            }
        
        elif detected_type == 'oral':
            # Oral cancer classification
            if ORAL_INFERENCE_AVAILABLE:
                classifier = get_oral_cancer_classifier()
                if classifier.model is not None:
                    result = classifier.predict(image)
                    result['detected_image_type'] = 'oral_histopathology'
                    return result
            
            # Demo mode for oral
            import random
            demo_types = [
                {'class': 'oral_normal', 'name': 'Normal Oral Tissue', 'severity': 'None',
                 'description': 'Normal oral tissue.',
                 'recommendation': 'No immediate action required.'},
                {'class': 'oral_scc', 'name': 'Oral Squamous Cell Carcinoma', 'severity': 'High',
                 'description': 'SCC detected in oral tissue.',
                 'recommendation': 'Urgent referral to oral surgeon/oncologist.'}
            ]
            demo = random.choice(demo_types)
            return {
                'status': 'demo',
                'message': 'Oral cancer model not loaded. Using demo response.',
                'detected_image_type': 'oral_histopathology',
                'prediction': {**demo, 'category': 'Oral Cancer', 'confidence': 0.88},
                'probabilities': {
                    'normal': 0.50,
                    'scc': 0.50
                }
            }
        
        else:
            # Breast cancer classification
            if BREAST_INFERENCE_AVAILABLE:
                classifier = get_breast_cancer_classifier()
                if classifier.model is not None:
                    result = classifier.predict(image)
                    result['detected_image_type'] = 'breast_histopathology'
                    return result
            
            # Demo mode for breast
            return {
                'status': 'demo',
                'message': 'Breast model not loaded. Using demo response.',
                'detected_image_type': 'breast_histopathology',
                'prediction': {
                    'class': 'breast_benign',
                    'name': 'Benign Breast Lesion',
                    'category': 'Breast Cancer',
                    'severity': 'Low',
                    'confidence': 0.92,
                    'description': 'Non-cancerous breast tissue.',
                    'recommendation': 'Regular monitoring recommended.'
                },
                'probabilities': {'benign': 0.92, 'malignant': 0.08}
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get('/predict/classes')
async def get_all_classes():
    """Get all supported cancer classes."""
    return {
        'breast_cancer': {
            'model_type': 'breast_cancer_binary',
            'classes': {
                'breast_benign': {'name': 'Benign Breast Lesion', 'severity': 'Low'},
                'breast_malignant': {'name': 'Malignant Breast Tumor', 'severity': 'High'}
            }
        },
        'brain_cancer': {
            'model_type': 'brain_cancer_multiclass',
            'classes': {
                'brain_glioma': {'name': 'Glioma Tumor', 'severity': 'High'},
                'brain_menin': {'name': 'Meningioma Tumor', 'severity': 'Medium'},
                'brain_tumor': {'name': 'Pituitary Tumor', 'severity': 'Medium'}
            }
        },
        'cervical_cancer': {
            'model_type': 'cervical_cancer_multiclass',
            'classes': {
                'cervix_dyk': {'name': 'Dyskeratotic Cells', 'severity': 'High'},
                'cervix_koc': {'name': 'Koilocytotic Cells', 'severity': 'Medium'},
                'cervix_mep': {'name': 'Metaplastic Cells', 'severity': 'Low'},
                'cervix_pab': {'name': 'Parabasal Cells', 'severity': 'Low'},
                'cervix_sfi': {'name': 'Superficial-Intermediate Cells', 'severity': 'None'}
            }
        },
        'kidney_cancer': {
            'model_type': 'kidney_cancer_binary',
            'classes': {
                'kidney_normal': {'name': 'Normal Kidney Tissue', 'severity': 'None'},
                'kidney_tumor': {'name': 'Kidney Tumor', 'severity': 'High'}
            }
        },
        'lung_cancer': {
            'model_type': 'lung_cancer_multiclass',
            'classes': {
                'lung_aca': {'name': 'Lung Adenocarcinoma', 'severity': 'High'},
                'lung_bnt': {'name': 'Benign Lung Tissue', 'severity': 'None'},
                'lung_scc': {'name': 'Squamous Cell Carcinoma', 'severity': 'High'}
            }
        },
        'colon_cancer': {
            'model_type': 'colon_cancer_binary',
            'classes': {
                'colon_aca': {'name': 'Colon Adenocarcinoma', 'severity': 'High'},
                'colon_bnt': {'name': 'Benign Colon Tissue', 'severity': 'None'}
            }
        },
        'lymphoma': {
            'model_type': 'lymphoma_multiclass',
            'classes': {
                'lymph_cll': {'name': 'Chronic Lymphocytic Leukemia', 'severity': 'High'},
                'lymph_fl': {'name': 'Follicular Lymphoma', 'severity': 'Medium'},
                'lymph_mcl': {'name': 'Mantle Cell Lymphoma', 'severity': 'High'}
            }
        },
        'oral_cancer': {
            'model_type': 'oral_cancer_binary',
            'classes': {
                'oral_normal': {'name': 'Normal Oral Tissue', 'severity': 'None'},
                'oral_scc': {'name': 'Oral Squamous Cell Carcinoma', 'severity': 'High'}
            }
        }
    }


@router.get('/predict/status')
async def get_model_status():
    """Get status of all models."""
    status = {
        'breast_cancer': {'status': 'unavailable', 'model_loaded': False},
        'brain_cancer': {'status': 'unavailable', 'model_loaded': False},
        'cervical_cancer': {'status': 'unavailable', 'model_loaded': False},
        'kidney_cancer': {'status': 'unavailable', 'model_loaded': False},
        'lung_cancer': {'status': 'unavailable', 'model_loaded': False},
        'colon_cancer': {'status': 'unavailable', 'model_loaded': False},
        'lymphoma': {'status': 'unavailable', 'model_loaded': False},
        'oral_cancer': {'status': 'unavailable', 'model_loaded': False}
    }
    
    if BREAST_INFERENCE_AVAILABLE:
        classifier = get_breast_cancer_classifier()
        model_loaded = classifier.model is not None
        status['breast_cancer'] = {
            'status': 'ready' if model_loaded else 'demo_mode',
            'model_loaded': model_loaded,
            'device': classifier.device if model_loaded else None
        }
    
    if BRAIN_INFERENCE_AVAILABLE:
        classifier = get_brain_cancer_classifier()
        model_loaded = classifier.model is not None
        status['brain_cancer'] = {
            'status': 'ready' if model_loaded else 'demo_mode',
            'model_loaded': model_loaded,
            'device': classifier.device if model_loaded else None
        }
    
    if CERVICAL_INFERENCE_AVAILABLE:
        classifier = get_cervical_cancer_classifier()
        model_loaded = classifier.model is not None
        status['cervical_cancer'] = {
            'status': 'ready' if model_loaded else 'demo_mode',
            'model_loaded': model_loaded,
            'device': classifier.device if model_loaded else None
        }
    
    if KIDNEY_INFERENCE_AVAILABLE:
        classifier = get_kidney_cancer_classifier()
        model_loaded = classifier.model is not None
        status['kidney_cancer'] = {
            'status': 'ready' if model_loaded else 'demo_mode',
            'model_loaded': model_loaded,
            'device': classifier.device if model_loaded else None
        }
    
    if LUNG_INFERENCE_AVAILABLE:
        classifier = get_lung_cancer_classifier()
        model_loaded = classifier.model is not None
        status['lung_cancer'] = {
            'status': 'ready' if model_loaded else 'demo_mode',
            'model_loaded': model_loaded,
            'device': classifier.device if model_loaded else None
        }
    
    if COLON_INFERENCE_AVAILABLE:
        classifier = get_colon_cancer_classifier()
        model_loaded = classifier.model is not None
        status['colon_cancer'] = {
            'status': 'ready' if model_loaded else 'demo_mode',
            'model_loaded': model_loaded,
            'device': classifier.device if model_loaded else None
        }
    
    if LYMPHOMA_INFERENCE_AVAILABLE:
        classifier = get_lymphoma_classifier()
        model_loaded = classifier.model is not None
        status['lymphoma'] = {
            'status': 'ready' if model_loaded else 'demo_mode',
            'model_loaded': model_loaded,
            'device': classifier.device if model_loaded else None
        }
    
    if ORAL_INFERENCE_AVAILABLE:
        classifier = get_oral_cancer_classifier()
        model_loaded = classifier.model is not None
        status['oral_cancer'] = {
            'status': 'ready' if model_loaded else 'demo_mode',
            'model_loaded': model_loaded,
            'device': classifier.device if model_loaded else None
        }
    
    return status

