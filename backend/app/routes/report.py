"""
OncoAI Fusion - Report Generation API Routes
AI-powered diagnostic report generation using LLM
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import logging

from app.services.llm_service import get_llm_generator

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()


class PredictionInput(BaseModel):
    """Input model for report generation."""
    cancer_type: str = Field(..., description="Cancer class ID (e.g., 'brain_glioma')")
    cancer_name: str = Field(default="", description="Human-readable cancer name")
    category: str = Field(default="", description="Cancer category")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    severity: str = Field(default="", description="Severity level")
    patient_age: Optional[int] = Field(default=None, ge=0, le=120, description="Patient age")
    patient_gender: Optional[str] = Field(default=None, description="Patient gender")
    additional_notes: Optional[str] = Field(default=None, description="Additional clinical notes")


class ReportOutput(BaseModel):
    """Output model for generated report."""
    status: str
    report_id: str
    generated_at: str
    report: dict
    llm_enhanced: bool = False  # Indicates if LLM was used for generation
    disclaimer: str


# Cancer information for report generation
CANCER_INFO = {
    'brain_glioma': {
        'name': 'Brain Glioma',
        'description': 'A type of tumor that occurs in the brain and spinal cord, originating from glial cells.',
        'typical_treatment': 'Surgery, radiation therapy, chemotherapy, targeted drug therapy',
        'prognosis_factors': 'Grade of tumor, location, patient age, overall health',
        'urgency': 'High - requires immediate specialist consultation'
    },
    'brain_menin': {
        'name': 'Brain Meningioma',
        'description': 'A tumor that arises from the meninges, the membranes surrounding the brain and spinal cord.',
        'typical_treatment': 'Observation, surgery, radiation therapy',
        'prognosis_factors': 'Tumor grade, size, location, completeness of surgical removal',
        'urgency': 'Medium - specialist consultation recommended'
    },
    'brain_tumor': {
        'name': 'Brain Tumor',
        'description': 'A mass or growth of abnormal cells in the brain.',
        'typical_treatment': 'Surgery, radiation therapy, chemotherapy, targeted therapy',
        'prognosis_factors': 'Type of tumor, grade, location, size, patient factors',
        'urgency': 'High - requires immediate specialist consultation'
    },
    'breast_benign': {
        'name': 'Breast Benign Lesion',
        'description': 'Non-cancerous breast tissue changes that do not spread to other parts of the body.',
        'typical_treatment': 'Monitoring, possible surgical removal if symptomatic',
        'prognosis_factors': 'Type of lesion, family history, hormonal factors',
        'urgency': 'Low - routine follow-up recommended'
    },
    'breast_malignant': {
        'name': 'Breast Malignant Tumor',
        'description': 'Cancerous cells in the breast tissue that can spread to other parts of the body.',
        'typical_treatment': 'Surgery, chemotherapy, radiation, hormonal therapy, targeted therapy',
        'prognosis_factors': 'Stage, grade, hormone receptor status, HER2 status',
        'urgency': 'High - immediate oncology consultation required'
    },
    'lung_aca': {
        'name': 'Lung Adenocarcinoma',
        'description': 'The most common type of lung cancer, usually found in the outer parts of the lung.',
        'typical_treatment': 'Surgery, chemotherapy, radiation, targeted therapy, immunotherapy',
        'prognosis_factors': 'Stage, molecular markers, overall health, response to treatment',
        'urgency': 'High - immediate pulmonology/oncology consultation required'
    },
    'lung_scc': {
        'name': 'Lung Squamous Cell Carcinoma',
        'description': 'A type of non-small cell lung cancer typically found in the central part of the lungs.',
        'typical_treatment': 'Surgery, chemotherapy, radiation, immunotherapy',
        'prognosis_factors': 'Stage, location, patient health, molecular markers',
        'urgency': 'High - immediate specialist consultation required'
    },
    'lung_bnt': {
        'name': 'Lung Benign Tissue',
        'description': 'Normal lung tissue without signs of malignancy.',
        'typical_treatment': 'No treatment required',
        'prognosis_factors': 'N/A - normal tissue',
        'urgency': 'None - routine follow-up as appropriate'
    },
    'colon_aca': {
        'name': 'Colon Adenocarcinoma',
        'description': 'Cancer that begins in the cells lining the colon.',
        'typical_treatment': 'Surgery, chemotherapy, targeted therapy, immunotherapy',
        'prognosis_factors': 'Stage, grade, molecular markers, lymph node involvement',
        'urgency': 'High - immediate gastroenterology/oncology consultation required'
    },
    'colon_bnt': {
        'name': 'Colon Benign Tissue',
        'description': 'Normal colon tissue without signs of malignancy.',
        'typical_treatment': 'No treatment required, regular screening recommended',
        'prognosis_factors': 'N/A - normal tissue',
        'urgency': 'None - continue routine screening'
    },
    'kidney_tumor': {
        'name': 'Kidney Tumor',
        'description': 'Abnormal growth in the kidney that may be malignant.',
        'typical_treatment': 'Surgery, targeted therapy, immunotherapy',
        'prognosis_factors': 'Type, stage, grade, overall health',
        'urgency': 'High - urology/oncology consultation required'
    },
    'kidney_normal': {
        'name': 'Kidney Normal',
        'description': 'Normal kidney tissue without pathological findings.',
        'typical_treatment': 'No treatment required',
        'prognosis_factors': 'N/A - normal tissue',
        'urgency': 'None - routine follow-up as appropriate'
    },
    'oral_scc': {
        'name': 'Oral Squamous Cell Carcinoma',
        'description': 'Cancer of the mouth and oral cavity.',
        'typical_treatment': 'Surgery, radiation, chemotherapy',
        'prognosis_factors': 'Stage, location, HPV status, depth of invasion',
        'urgency': 'High - immediate ENT/oncology consultation required'
    },
    'oral_normal': {
        'name': 'Oral Normal',
        'description': 'Normal oral tissue without pathological findings.',
        'typical_treatment': 'No treatment required',
        'prognosis_factors': 'N/A - normal tissue',
        'urgency': 'None - routine dental follow-up'
    },
}

# Add remaining classes with generic info
for class_id in ['cervix_dyk', 'cervix_koc', 'cervix_mep', 'cervix_pab', 'cervix_sfi',
                  'lymph_cll', 'lymph_fl', 'lymph_mcl']:
    if class_id not in CANCER_INFO:
        category = 'Cervical Cancer' if class_id.startswith('cervix') else 'Lymphoma'
        CANCER_INFO[class_id] = {
            'name': class_id.replace('_', ' ').title(),
            'description': f'A type of {category.lower()} identified through pathological analysis.',
            'typical_treatment': 'Specialist consultation required for treatment planning',
            'prognosis_factors': 'Stage, grade, patient factors, response to treatment',
            'urgency': 'Medium to High - specialist consultation recommended'
        }


def generate_report_content(data: PredictionInput) -> tuple[dict, bool]:
    """
    Generate a comprehensive diagnostic report.
    
    Uses LLM for enhanced generation with graceful fallback to template-based reports.
    
    Returns:
        tuple: (report_dict, llm_enhanced_bool)
    """
    
    # Get LLM generator
    llm_generator = get_llm_generator()
    llm_enhanced = False
    llm_content = None
    
    # Try LLM generation first
    if llm_generator.is_available():
        logger.info("Attempting LLM-powered report generation...")
        prediction_data = {
            'cancer_type': data.cancer_type,
            'cancer_name': data.cancer_name,
            'category': data.category,
            'confidence': data.confidence,
            'severity': data.severity,
            'patient_age': data.patient_age,
            'patient_gender': data.patient_gender,
            'additional_notes': data.additional_notes
        }
        llm_content = llm_generator.generate_diagnostic_report(prediction_data)
        if llm_content:
            llm_enhanced = True
            logger.info("✅ LLM report generation successful")
        else:
            logger.info("⚠️ LLM generation failed, using template fallback")
    else:
        logger.info("ℹ️ LLM not available, using template-based report")
    
    # Get cancer info for template sections
    cancer_info = CANCER_INFO.get(data.cancer_type, {
        'name': data.cancer_name or data.cancer_type,
        'description': 'Cancer type identified through AI analysis.',
        'typical_treatment': 'Consult with oncology specialist',
        'prognosis_factors': 'Multiple factors to be evaluated by specialist',
        'urgency': 'Specialist consultation recommended'
    })
    
    confidence_level = (
        'Very High' if data.confidence >= 0.95 else
        'High' if data.confidence >= 0.85 else
        'Moderate' if data.confidence >= 0.70 else
        'Low' if data.confidence >= 0.50 else
        'Very Low'
    )
    
    # Build report sections
    report = {
        'header': {
            'title': 'OncoAI Fusion Diagnostic Analysis Report',
            'report_type': 'AI-Assisted Preliminary Analysis',
            'analysis_date': datetime.utcnow().isoformat(),
            'ai_system_version': '1.0.0',
            'generation_method': 'LLM-Enhanced' if llm_enhanced else 'Template-Based'
        },
        'patient_info': {
            'age': data.patient_age,
            'gender': data.patient_gender,
            'notes': data.additional_notes
        },
        'diagnosis': {
            'primary_finding': cancer_info['name'],
            'classification_code': data.cancer_type,
            'category': data.category or 'Cancer',
            'severity': data.severity or 'To be determined',
            'confidence_score': f"{data.confidence * 100:.1f}%",
            'confidence_level': confidence_level
        },
        'clinical_information': {
            'description': cancer_info['description'],
            'typical_treatment_options': cancer_info['typical_treatment'],
            'prognosis_factors': cancer_info['prognosis_factors'],
            'urgency': cancer_info['urgency']
        },
        'recommendations': {
            'immediate_actions': _get_immediate_actions(data, cancer_info),
            'follow_up': _get_follow_up_recommendations(data, cancer_info),
            'additional_tests': _get_additional_tests(data)
        },
        'ai_analysis_notes': {
            'model_type': 'ResNet50 Multi-Cancer Classifier',
            'analysis_method': 'Deep learning-based image classification',
            'training_data': 'Multi-institutional cancer histopathology dataset',
            'limitations': [
                'AI analysis is for research purposes only',
                'Results should be confirmed by qualified pathologist',
                'Clinical correlation is required',
                'Does not replace professional medical diagnosis'
            ]
        }
    }
    
    # Add LLM-generated content if available
    if llm_content:
        report['llm_analysis'] = {
            'clinical_summary': llm_content.get('clinical_summary', ''),
            'detailed_findings': llm_content.get('detailed_findings', ''),
            'patient_communication': llm_content.get('patient_communication', ''),
            'key_observations': llm_content.get('key_observations', []),
            'clinical_correlation': llm_content.get('clinical_correlation', '')
        }
    
    return report, llm_enhanced



def _get_immediate_actions(data: PredictionInput, cancer_info: dict) -> List[str]:
    """Get immediate action recommendations."""
    actions = []
    
    if 'High' in cancer_info.get('urgency', ''):
        actions.append('Schedule immediate consultation with specialist oncologist')
        actions.append('Request expedited pathological confirmation')
    elif 'Medium' in cancer_info.get('urgency', ''):
        actions.append('Schedule consultation with appropriate specialist')
        actions.append('Obtain pathological confirmation')
    else:
        actions.append('Continue routine monitoring')
        actions.append('Consider follow-up imaging if clinically indicated')
    
    if data.confidence < 0.70:
        actions.append('Consider additional imaging or biopsy due to lower confidence score')
    
    return actions


def _get_follow_up_recommendations(data: PredictionInput, cancer_info: dict) -> List[str]:
    """Get follow-up recommendations."""
    recommendations = [
        'Review results with treating physician',
        'Maintain complete medical documentation',
        'Monitor for any new symptoms'
    ]
    
    if 'malignant' in data.cancer_type or 'tumor' in data.cancer_type or 'scc' in data.cancer_type or 'aca' in data.cancer_type:
        recommendations.insert(0, 'Regular oncology follow-up appointments')
        recommendations.append('Consider genetic counseling if appropriate')
    
    return recommendations


def _get_additional_tests(data: PredictionInput) -> List[str]:
    """Get recommended additional tests."""
    tests = ['Histopathological confirmation by certified pathologist']
    
    if 'brain' in data.cancer_type:
        tests.extend(['MRI with contrast', 'Neurological assessment'])
    elif 'breast' in data.cancer_type:
        tests.extend(['Mammogram', 'Ultrasound', 'Hormone receptor testing'])
    elif 'lung' in data.cancer_type:
        tests.extend(['CT scan', 'PET scan', 'Pulmonary function tests', 'Molecular testing'])
    elif 'colon' in data.cancer_type:
        tests.extend(['Colonoscopy', 'CT scan', 'CEA tumor marker'])
    elif 'kidney' in data.cancer_type:
        tests.extend(['CT urogram', 'Renal function tests'])
    elif 'lymph' in data.cancer_type:
        tests.extend(['Flow cytometry', 'Bone marrow biopsy', 'PET scan'])
    elif 'oral' in data.cancer_type:
        tests.extend(['Oral examination', 'Biopsy', 'CT/MRI of head and neck'])
    elif 'cervix' in data.cancer_type:
        tests.extend(['Pap smear', 'HPV testing', 'Colposcopy'])
    
    return tests


@router.post('/generate_report', response_model=ReportOutput)
async def generate_report(data: PredictionInput):
    """
    Generate AI-powered diagnostic report based on prediction results.
    
    - **cancer_type**: The predicted cancer class ID
    - **confidence**: Prediction confidence score (0-1)
    - **patient_age**: Optional patient age
    - **patient_gender**: Optional patient gender
    - **additional_notes**: Optional clinical notes
    
    Returns a comprehensive diagnostic report with recommendations.
    """
    try:
        # Generate report (returns tuple of report dict and llm_enhanced flag)
        report, llm_enhanced = generate_report_content(data)
        
        # Create report ID
        report_id = f"ONCO-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{hash(data.cancer_type) % 10000:04d}"
        
        return ReportOutput(
            status='success',
            report_id=report_id,
            generated_at=datetime.utcnow().isoformat(),
            report=report,
            llm_enhanced=llm_enhanced,
            disclaimer=(
                "⚠️ IMPORTANT DISCLAIMER: This report is generated by an AI system for research "
                "and educational purposes only. It is NOT a clinical diagnosis and should NOT be "
                "used as the sole basis for medical decisions. All findings must be confirmed by "
                "qualified healthcare professionals. This system is not FDA approved for clinical use."
            )
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get('/report/template')
async def get_report_template():
    """Get empty report template for reference."""
    return {
        'required_fields': {
            'cancer_type': 'string - Cancer class ID',
            'confidence': 'float - Prediction confidence (0-1)'
        },
        'optional_fields': {
            'cancer_name': 'string - Human-readable name',
            'category': 'string - Cancer category',
            'severity': 'string - Severity level',
            'patient_age': 'integer - Patient age (0-120)',
            'patient_gender': 'string - Patient gender',
            'additional_notes': 'string - Clinical notes'
        },
        'example': {
            'cancer_type': 'brain_glioma',
            'cancer_name': 'Brain Glioma',
            'category': 'Brain Cancer',
            'confidence': 0.92,
            'severity': 'High',
            'patient_age': 45,
            'patient_gender': 'Male',
            'additional_notes': 'Patient presents with persistent headaches'
        }
    }
