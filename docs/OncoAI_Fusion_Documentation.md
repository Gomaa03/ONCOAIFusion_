# OncoAI Fusion - Comprehensive Project Documentation

---

## 1. PROJECT OVERVIEW

### Project Name & Tagline
**OncoAI Fusion** - *Multi-Cancer Diagnostic AI System*

### Purpose & Problem Solved
OncoAI Fusion addresses the challenge of rapid, preliminary cancer screening from medical imagery. It provides:
- **Automated image type detection** (Brain MRI vs Breast Histopathology)
- **Real-time cancer classification** with confidence scores
- **Clinical recommendations** and severity assessment
- **AI-generated diagnostic reports**

### Target Audience
- Medical researchers
- Healthcare professionals (for research/second-opinion purposes)
- Pathology labs
- Medical education institutions

### Current Status
**Version**: 1.0.0  
**Status**: Beta/Research  
**Disclaimer**: For research and educational purposes only. Not FDA approved for clinical use.

---

## 2. ARCHITECTURE & TECHNICAL DESIGN

### High-Level Architecture
```
┌─────────────┐     ┌─────────────────────────────────────────┐
│    User     │     │           DOCKER COMPOSE                │
│  (Browser)  │────▶│  ┌─────────────┐   ┌─────────────────┐  │
└─────────────┘     │  │  Frontend   │──▶│    Backend      │  │
                    │  │ (Nginx:80)  │   │ (FastAPI:8001)  │  │
                    │  └─────────────┘   └────────┬────────┘  │
                    │                             │           │
                    │            ┌────────────────┴───────────┤
                    │            ▼                            │
                    │  ┌─────────────────────────────────┐    │
                    │  │       ML Models (PyTorch)       │    │
                    │  │  Brain Classifier │ Breast Cls. │    │
                    │  └─────────────────────────────────┘    │
                    └─────────────────────────────────────────┘
```

### Technology Stack

| Layer | Technology | Version |
|-------|------------|---------|
| **Frontend** | React | 19.2.0 |
| **Build Tool** | Vite | 7.2.4 |
| **Backend** | FastAPI | 0.104.0 |
| **ML Framework** | PyTorch | 2.0.0 |
| **Computer Vision** | TorchVision | 0.15.0 |
| **ASGI Server** | Uvicorn | 0.24.0 |
| **Web Server** | Nginx | (Docker) |
| **Containerization** | Docker Compose | 3.8 |
| **Language** | Python 3.x, JavaScript ES6+ |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/predict` | POST | Unified cancer prediction (auto-detects image type) |
| `/api/v1/predict/classes` | GET | List supported cancer classes |
| `/api/v1/predict/status` | GET | Model loading status |
| `/api/v1/generate_report` | POST | Generate AI diagnostic report |
| `/api/v1/report/template` | GET | Report template reference |

### Authentication
- **Current**: Open API with CORS configured (`allow_origins=['*']`)
- **Production recommendation**: Add JWT/OAuth2 authentication

### Deployment Architecture
- **Docker Compose** orchestration with 2 containers
- **Frontend**: Nginx serving React build (port 80 → exposed 3000)
- **Backend**: Uvicorn serving FastAPI (port 8001)
- **Volumes**: Model checkpoints mounted as read-only
- **Health checks**: Configured for backend container

---

## 3. CODEBASE STRUCTURE

```
oncoai-fusion/
├── backend/                      # FastAPI Backend
│   ├── main.py                   # Application entry point
│   ├── requirements.txt          # Python dependencies
│   ├── app/
│   │   ├── routes/
│   │   │   ├── health.py         # Health check endpoint
│   │   │   ├── predict.py        # Prediction API (208 lines)
│   │   │   └── report.py         # Report generation (341 lines)
│   │   ├── services/
│   │   │   ├── brain_cancer_inference.py   # Brain classifier
│   │   │   ├── breast_cancer_inference.py  # Breast classifier
│   │   │   └── inference.py      # Multi-cancer inference
│   │   ├── models/               # Pydantic models
│   │   ├── config/               # Configuration
│   │   └── utils/                # Utilities
│   └── tests/                    # Unit tests
│
├── frontend/                     # React Frontend
│   ├── package.json              # npm dependencies
│   ├── vite.config.js            # Vite configuration
│   ├── index.html                # HTML entry
│   └── src/
│       ├── App.jsx               # Main component (492 lines)
│       ├── App.css               # Styles
│       └── main.jsx              # React entry
│
├── ml_models/                    # Machine Learning
│   ├── training/
│   │   ├── train.py              # Multi-cancer training
│   │   ├── train_brain_cancer.py # Brain model training
│   │   └── train_breast_cancer.py # Breast model training (366 lines)
│   ├── checkpoints/              # Saved model weights
│   │   ├── brain_cancer/best_model.pth
│   │   └── breast_cancer/best_model.pth
│   ├── evaluation/               # Evaluation scripts
│   └── logs/                     # Training logs
│
├── deployment/
│   ├── docker/
│   │   ├── docker-compose.yml    # Container orchestration
│   │   ├── Dockerfile.backend    # Backend image
│   │   └── Dockerfile.frontend   # Frontend image
│   ├── kubernetes/               # K8s configs (future)
│   └── scripts/                  # Deployment scripts
│
├── data/                         # Training data (external)
├── docs/                         # Documentation
├── README.md                     # Project readme
└── .env                          # Environment variables
```

### Key Design Patterns
- **Singleton Pattern**: Global model instance (`_classifier`)
- **Factory Pattern**: `get_breast_cancer_classifier()` model creation
- **Strategy Pattern**: Image type detection routes to appropriate model
- **Repository Pattern**: Separation of routes/services

---

## 4. FUNCTIONALITY & FEATURES

### Core Features

#### 1. Automatic Image Type Detection
```python
def detect_image_type(image: Image.Image) -> str:
    # Analyzes pixel color distribution
    # Brain MRI: Grayscale (R≈G≈B)
    # Breast Histopathology: Pink/purple staining
```

#### 2. Brain Cancer Classification (3 classes)
- **Glioma** - High severity
- **Meningioma** - Medium severity
- **Pituitary Tumor** - Medium severity

#### 3. Breast Cancer Classification (2 classes)
- **Benign** - Low severity
- **Malignant** - High severity

#### 4. AI Diagnostic Report Generation
- Patient information
- Diagnosis details with confidence
- Clinical recommendations
- Additional test suggestions

### User Workflow
1. User uploads medical image (drag-drop or file picker)
2. Frontend sends image to `/api/v1/predict`
3. Backend auto-detects image type
4. Appropriate model classifies image
5. JSON response with prediction, confidence, severity
6. User can request detailed report

### Input/Output Specifications

**Input:**
- Image formats: JPG, PNG, BMP
- Preprocessing: Resize to 224×224, normalize (ImageNet stats)

**Output:**
```json
{
  "status": "success",
  "detected_image_type": "brain_mri",
  "prediction": {
    "class": "brain_glioma",
    "name": "Glioma Tumor",
    "severity": "High",
    "confidence": 0.92,
    "recommendation": "..."
  },
  "probabilities": {
    "glioma": 0.92,
    "meningioma": 0.05,
    "pituitary": 0.03
  }
}
```

### Error Handling
- Invalid file type → 400 Bad Request
- Model not loaded → Demo mode response
- Server error → 500 with error detail

---

## 5. DEPENDENCIES & INTEGRATIONS

### Backend Dependencies (requirements.txt)
```
fastapi==0.104.0
uvicorn==0.24.0
torch==2.0.0
torchvision==0.15.0
Pillow==10.0.0
numpy==1.24.0
pandas==2.0.0
scikit-learn==1.3.0
openai==1.0.0
langchain==0.0.350
pydantic==2.0.0
python-dotenv==1.0.0
```

### Frontend Dependencies (package.json)
```json
{
  "dependencies": {
    "axios": "^1.13.2",
    "lucide-react": "^0.555.0",
    "react": "^19.2.0",
    "react-dom": "^19.2.0"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^5.1.1",
    "eslint": "^9.39.1",
    "vite": "^7.2.4"
  }
}
```

### Third-Party Integrations
- **OpenAI/LangChain**: Prepared for LLM-powered report enhancement
- **ImageNet**: Pre-trained weights for transfer learning

---

## 6. DEVELOPMENT WORKFLOW

### Environment Setup

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python main.py  # Runs on port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev  # Runs on port 3000
```

**Docker:**
```bash
cd deployment/docker
docker-compose up --build
# Frontend: http://localhost:3000
# Backend: http://localhost:8001
```

### Build Process
```bash
# Frontend production build
npm run build

# Backend - no build step (Python)

# Docker images
docker-compose build
```

### Code Quality Tools
- **ESLint**: JavaScript linting (`npm run lint`)
- **Type hints**: Python type annotations throughout

---

## 7. CURRENT STATE & METRICS

### Codebase Size
| Metric | Value |
|--------|-------|
| Total source files | ~25 |
| Backend Python | ~1,500+ lines |
| Frontend JavaScript | ~600+ lines |
| Training scripts | ~1,000+ lines |

### Model Architecture
| Model | Classes | Architecture | Input Size |
|-------|---------|--------------|------------|
| Brain Cancer | 3 | ResNet50 + Custom Head | 224×224 |
| Breast Cancer | 2 | ResNet50 + Custom Head | 224×224 |

### Performance
- **Inference time**: <100ms
- **Device support**: CPU, CUDA, Apple MPS

---

## 8. TECHNICAL DECISIONS & RATIONALE

### ResNet50 Selection
**Decision**: Use ResNet50 pre-trained on ImageNet  
**Rationale**: 
- Excellent balance of accuracy and inference speed
- Well-suited for medical imaging transfer learning
- Widely validated in research

### Custom Classification Head
```python
nn.Sequential(
    nn.Linear(2048, 256),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(256),
    nn.Dropout(0.5),
    nn.Linear(256, num_classes)
)
```
**Rationale**: Dropout prevents overfitting on limited medical data

### Separate Models vs Multi-task
**Decision**: Separate models for brain and breast cancer  
**Rationale**: 
- Different image modalities (MRI vs histopathology)
- Easier independent training and updates
- Better accuracy per domain

### FastAPI over Flask
**Decision**: Use FastAPI  
**Rationale**: 
- Async support for concurrent requests
- Automatic OpenAPI documentation
- Pydantic validation

---

## 9. FUTURE ROADMAP

### Planned Features
1. **Additional cancer types**: Lung, Colon, Kidney, Cervical, Lymphoma, Oral
2. **LLM-powered reports**: Enhanced natural language generation
3. **Explainable AI**: Grad-CAM visualization of model attention
4. **Multi-language support**: Internationalization

### Technical Improvements
- Kubernetes deployment for auto-scaling
- Model versioning and A/B testing
- Real-time model retraining pipeline
- Authentication and user management

### Scalability Considerations
- Model serving with TorchServe or Triton
- Redis caching for predictions
- Load balancing across GPU instances

---

## 10. SECURITY & COMPLIANCE

### Current Security Measures
- ✅ CORS middleware configured
- ✅ Input validation with Pydantic
- ✅ Docker containerization (isolated execution)
- ✅ Read-only volume mounts for models

### Recommendations for Production
- Add JWT/OAuth2 authentication
- Implement rate limiting
- HTTPS with SSL certificates
- Audit logging for predictions
- HIPAA compliance for patient data

### Data Privacy
- **No data storage**: Images processed in-memory only
- **No patient data**: Optional fields in report generation
- **Disclaimer**: Clear research-only usage statements

---

## 11. DOCUMENTATION & RESOURCES

### Existing Documentation
- **README.md**: Quick start guide, project structure
- **Swagger UI**: Auto-generated at `/docs`
- **Code comments**: Docstrings throughout

### API Documentation
- FastAPI auto-generates OpenAPI spec
- Access at: `http://localhost:8000/docs`

### Onboarding New Developers
1. Clone repository
2. Follow README setup instructions
3. Review `backend/app/routes/predict.py` for API logic
4. Review `ml_models/training/` for model architecture
5. Run with Docker for full stack testing

---

## Quick Reference

| Resource | Location |
|----------|----------|
| Backend Entry | `backend/main.py` |
| Prediction Logic | `backend/app/routes/predict.py` |
| Brain Model | `backend/app/services/brain_cancer_inference.py` |
| Breast Model | `backend/app/services/breast_cancer_inference.py` |
| Training | `ml_models/training/train_*.py` |
| Model Weights | `ml_models/checkpoints/*/best_model.pth` |
| Frontend App | `frontend/src/App.jsx` |
| Docker Config | `deployment/docker/docker-compose.yml` |

---

> **Generated**: December 2024  
> **Project**: OncoAI Fusion v1.0.0
