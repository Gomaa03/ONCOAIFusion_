import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    DATASET_ROOT = os.getenv('ONCOAI_DATASET_ROOT', '/Users/lekhans/Desktop/new folder/dataset_new')
    MODEL_PATH = os.getenv('MODEL_PATH', './ml_models/checkpoints')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    CANCER_TYPES = ['brain_glioma', 'brain_menin', 'brain_tumor', 'breast_benign', 'breast_malignant', 'cervix_dyk', 'cervix_koc', 'cervix_mep', 'cervix_pab', 'cervix_sfi', 'colon_aca', 'colon_bnt', 'kidney_normal', 'kidney_tumor', 'lung_aca', 'lung_bnt', 'lung_scc', 'lymph_cll', 'lymph_fl', 'lymph_mcl', 'oral_normal', 'oral_scc']
    BATCH_SIZE = 32
    IMAGE_SIZE = 224

settings = Settings()
