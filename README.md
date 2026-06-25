# OncoAI Fusion

Multi-cancer diagnostic AI system using deep learning and generative AI.

## 🎯 Overview

OncoAI Fusion is a comprehensive multi-cancer diagnostic system that combines:
- **Deep Learning**: ResNet50-based classification for 22 cancer subtypes
- **Generative AI**: LLM-powered report generation
- **Agentic Reasoning**: Intelligent diagnostic workflows

## 📊 Supported Cancer Types

| Category | Subtypes |
|----------|----------|
| Brain Cancer | glioma, meningioma, tumor |
| Breast Cancer | benign, malignant |
| Cervical Cancer | dyk, koc, mep, pab, sfi |
| Kidney Cancer | normal, tumor |
| Lung & Colon | lung_aca, lung_bnt, lung_scc, colon_aca, colon_bnt |
| Lymphoma | cll, fl, mcl |
| Oral Cancer | normal, scc |

## 🚀 Quick Start

### Backend
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend
```bash
cd frontend
npm install
npm start
```

### Docker
```bash
cd deployment/docker
docker-compose up --build
```

## 📁 Project Structure

```
oncoai-fusion/
├── backend/          # FastAPI backend
│   ├── app/
│   │   ├── models/   # ML model definitions
│   │   ├── routes/   # API endpoints
│   │   ├── services/ # Business logic
│   │   ├── config/   # Configuration
│   │   └── utils/    # Utilities
│   └── main.py
├── frontend/         # React frontend
│   └── src/
├── ml_models/        # Deep learning models
│   ├── training/     # Training scripts
│   ├── evaluation/   # Metrics & evaluation
│   └── checkpoints/  # Saved models
├── data/             # Training data
├── deployment/       # Docker & K8s configs
└── docs/             # Documentation
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/predict` | POST | Cancer classification |
| `/api/v1/generate_report` | POST | Generate diagnostic report

- **Training**: 77,000 images (22 classes)
- **Validation**: 22,000 images
- **Test**: 11,002 images

## 📚 API Documentation

Swagger UI: http://localhost:8000/docs

## ⚠️ Disclaimer

This system is for research purposes only. Not intended for clinical diagnosis.
