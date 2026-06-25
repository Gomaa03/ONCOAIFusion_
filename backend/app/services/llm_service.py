"""
OncoAI Fusion - LLM Service for Diagnostic Report Generation
Uses Groq API (FREE) for generating clinically-structured diagnostic reports
Fallback: Template-based reports when API is unavailable
"""

import os
import json
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import groq
GROQ_AVAILABLE = False
try:
    from groq import Groq
    GROQ_AVAILABLE = True
    logger.info("Using Groq library")
except ImportError:
    logger.warning("⚠️ Groq not installed. Run: pip install groq")


class LLMReportGenerator:
    """
    LLM-powered diagnostic report generator using Groq (FREE Llama models).
    Provides graceful fallback when API is unavailable.
    """
    
    def __init__(self):
        """Initialize the Groq client if API key is available."""
        self.api_key = os.getenv('GROQ_API_KEY')
        self.model_name = os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')
        self.client = None
        
        if not GROQ_AVAILABLE:
            logger.warning("⚠️ Groq library not available")
            return
            
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info(f"✅ Groq client initialized successfully (model: {self.model_name})")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize Groq client: {e}")
                self.client = None
        else:
            logger.info("ℹ️ No Groq API key configured - using template-based reports")
    
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return self.client is not None
    
    def _build_clinical_prompt(self, prediction_data: Dict[str, Any]) -> str:
        """Build a clinical prompt for diagnostic report generation."""
        
        cancer_type = prediction_data.get('cancer_type', 'Unknown')
        cancer_name = prediction_data.get('cancer_name', cancer_type.replace('_', ' ').title())
        confidence = prediction_data.get('confidence', 0) * 100
        severity = prediction_data.get('severity', 'To be determined')
        category = prediction_data.get('category', 'Cancer')
        patient_age = prediction_data.get('patient_age')
        patient_gender = prediction_data.get('patient_gender')
        additional_notes = prediction_data.get('additional_notes', '')
        
        patient_info = ""
        if patient_age or patient_gender:
            patient_info = f"Patient: {patient_age or 'Unknown'} year old {patient_gender or 'patient'}"
        
        prompt = f"""You are a medical AI assistant helping to generate a preliminary diagnostic analysis report. 
This is for RESEARCH and EDUCATIONAL purposes only - NOT for clinical decision making.

Based on the AI image analysis results, generate a structured diagnostic report.

## AI Analysis Results:
- **Classification**: {cancer_name}
- **Classification Code**: {cancer_type}
- **Category**: {category}
- **AI Confidence Score**: {confidence:.1f}%
- **Indicated Severity**: {severity}
{f'- **{patient_info}**' if patient_info else ''}
{f'- **Clinical Notes**: {additional_notes}' if additional_notes else ''}

## Instructions:
Generate a JSON response with the following structure. Return ONLY the JSON, no markdown code blocks or extra text:
{{
    "clinical_summary": "A concise 2-3 sentence clinical summary of the AI findings, written professionally.",
    "detailed_findings": "A detailed paragraph describing the classification, what it typically indicates, and relevant clinical considerations.",
    "patient_communication": "A simplified, empathetic explanation suitable for communicating with patients (2-3 sentences, avoiding overly technical language).",
    "key_observations": ["List of 3-4 key observations from the analysis"],
    "clinical_correlation": "Brief note on how these findings should be correlated with clinical presentation and additional testing."
}}

Important:
- Be factual and professional
- Emphasize this is AI-assisted preliminary analysis only
- Do not provide definitive diagnoses
- Recommend specialist consultation where appropriate
- Return ONLY valid JSON"""

        return prompt
    
    def generate_diagnostic_report(self, prediction_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate an LLM-powered diagnostic report using Groq.
        
        Args:
            prediction_data: Dictionary containing prediction results
            
        Returns:
            Dictionary with LLM-generated report sections, or None if generation fails
        """
        if not self.is_available():
            logger.info("LLM not available, skipping generation")
            return None
        
        try:
            prompt = self._build_clinical_prompt(prediction_data)
            
            # Generate response using Groq
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical AI report assistant. Generate accurate, professional diagnostic reports. Always respond with valid JSON only, no markdown formatting."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up response if wrapped in markdown code blocks
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()
            
            # Parse the response
            llm_report = json.loads(content)
            
            logger.info("✅ Groq LLM report generated successfully")
            return llm_report
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse LLM response as JSON: {e}")
            return None
        except Exception as e:
            error_msg = str(e)
            if 'rate' in error_msg.lower() or '429' in error_msg:
                logger.warning(f"⚠️ Rate limit reached: {e}")
            else:
                logger.error(f"❌ Groq API error: {e}")
            return None


# Singleton instance
_llm_generator: Optional[LLMReportGenerator] = None


def get_llm_generator() -> LLMReportGenerator:
    """Get or create the LLM generator singleton."""
    global _llm_generator
    if _llm_generator is None:
        _llm_generator = LLMReportGenerator()
    return _llm_generator
