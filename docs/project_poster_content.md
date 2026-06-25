# OncoAI Fusion - Project Poster Content

---

## 📋 ABSTRACT

**OncoAI Fusion** is an intelligent multi-cancer diagnostic system leveraging deep learning for automated classification of medical imagery. The system employs **ResNet50-based convolutional neural networks** to analyze **Brain MRI scans** and **Breast Histopathology images**, providing real-time classification with confidence scores, severity assessment, and clinical recommendations.

The platform features **automatic image type detection** that identifies whether an uploaded image is a brain MRI or breast histopathology sample, then routes it to the appropriate specialized classifier. For brain cancer, the system classifies images into three categories: **Glioma, Meningioma, and Pituitary tumors**. For breast cancer, it performs binary classification between **Benign and Malignant** lesions.

Built on a modern microservices architecture with a **FastAPI backend** and **React frontend**, OncoAI Fusion demonstrates the practical application of transfer learning in medical imaging, achieving high accuracy while maintaining real-time inference capabilities. The system is containerized using **Docker** for easy deployment and scalability.

---

## 🎯 OBJECTIVES

1. **Develop an automated multi-cancer detection system** capable of classifying brain tumors (Glioma, Meningioma, Pituitary) and breast lesions (Benign, Malignant) from medical images.

2. **Implement automatic image type detection** to identify whether an uploaded image is a Brain MRI or Breast Histopathology sample without user specification.

3. **Achieve high classification accuracy** using transfer learning with pre-trained ResNet50 architecture fine-tuned on domain-specific medical imaging datasets.

4. **Build a user-friendly web interface** that provides intuitive image upload, real-time analysis visualization, and comprehensive diagnostic reports.

5. **Provide clinical decision support** through severity assessment, probability distributions, and actionable medical recommendations.

6. **Create a scalable, production-ready system** using containerized microservices architecture for easy deployment in clinical and research environments.

---

## 🔬 METHODOLOGY

### Data Collection & Preprocessing
- **Brain Cancer Dataset**: MRI scans categorized into Glioma, Meningioma, and Pituitary tumor classes
- **Breast Cancer Dataset**: Histopathology images classified as Benign or Malignant
- **Image Preprocessing**: 
  - Resize to 224×224 pixels
  - Normalization using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  - Data augmentation during training (rotation, flipping, color jittering)

### Model Architecture
- **Base Model**: ResNet50 pre-trained on ImageNet
- **Transfer Learning**: Feature extraction using frozen backbone layers with custom classification head
- **Classification Head**:
  - Fully connected layer (2048 → 256)
  - Batch Normalization + ReLU + Dropout (0.5)
  - Output layer (256 → num_classes)

### Training Process
- **Optimizer**: Adam with learning rate scheduling
- **Loss Function**: Cross-Entropy Loss
- **Regularization**: Dropout (0.5), Early stopping, Model checkpointing
- **Validation**: Training/Validation/Test split

### System Architecture
1. **Frontend (React + Vite)**: Interactive UI with drag-drop upload, real-time visualization
2. **Backend (FastAPI + Python)**: RESTful API, model inference, image processing
3. **ML Models (PyTorch)**: Separate classifiers for brain and breast cancer
4. **Auto-Detection**: Feature-based classification to route images to appropriate model

---

## 🏗️ MODEL ARCHITECTURE DIAGRAM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OncoAI Fusion Architecture                         │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────┐
                              │   User Upload    │
                              │  (Brain MRI /    │
                              │  Breast Histop.) │
                              └────────┬─────────┘
                                       │
                              ┌────────▼─────────┐
                              │ React Frontend   │
                              │ (Image Preview,  │
                              │  Upload Handler) │
                              └────────┬─────────┘
                                       │ HTTP POST
                              ┌────────▼─────────┐
                              │ FastAPI Backend  │
                              │ /api/v1/predict  │
                              └────────┬─────────┘
                                       │
                              ┌────────▼─────────┐
                              │  Auto-Detection  │
                              │ (Image Type ID)  │
                              └────────┬─────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                      │
           ┌────────▼────────┐                   ┌────────▼────────┐
           │  Brain Cancer   │                   │ Breast Cancer   │
           │   Classifier    │                   │   Classifier    │
           └────────┬────────┘                   └────────┬────────┘
                    │                                      │
    ┌───────────────┴───────────────┐        ┌────────────┴────────────┐
    │       ResNet50 Backbone       │        │    ResNet50 Backbone    │
    │    (ImageNet Pre-trained)     │        │  (ImageNet Pre-trained) │
    └───────────────┬───────────────┘        └────────────┬────────────┘
                    │                                      │
    ┌───────────────┴───────────────┐        ┌────────────┴────────────┐
    │   Custom Classification Head  │        │ Custom Classification   │
    │  ┌─────────────────────────┐  │        │         Head            │
    │  │ FC: 2048 → 256          │  │        │ ┌─────────────────────┐ │
    │  │ BatchNorm + ReLU        │  │        │ │ FC: 2048 → 256      │ │
    │  │ Dropout(0.5)            │  │        │ │ BatchNorm + ReLU    │ │
    │  │ FC: 256 → 3 classes     │  │        │ │ Dropout(0.5)        │ │
    │  └─────────────────────────┘  │        │ │ FC: 256 → 2 classes │ │
    └───────────────┬───────────────┘        │ └─────────────────────┘ │
                    │                        └────────────┬────────────┘
                    │                                      │
    ┌───────────────┴───────────────┐        ┌────────────┴────────────┐
    │         Softmax Output        │        │     Softmax Output      │
    │  ┌─────────────────────────┐  │        │ ┌─────────────────────┐ │
    │  │ • Glioma      (prob %)  │  │        │ │ • Benign   (prob %) │ │
    │  │ • Meningioma  (prob %)  │  │        │ │ • Malignant(prob %) │ │
    │  │ • Pituitary   (prob %)  │  │        │ └─────────────────────┘ │
    │  └─────────────────────────┘  │        └────────────┬────────────┘
    └───────────────┬───────────────┘                     │
                    │                                      │
                    └──────────────────┬───────────────────┘
                                       │
                              ┌────────▼─────────┐
                              │  JSON Response   │
                              │ • Prediction     │
                              │ • Confidence     │
                              │ • Severity       │
                              │ • Recommendation │
                              └────────┬─────────┘
                                       │
                              ┌────────▼─────────┐
                              │ React UI Display │
                              │ • Result Card    │
                              │ • Probability    │
                              │   Distribution   │
                              │ • Clinical Rec.  │
                              └──────────────────┘
```

---

## 📊 RESULTS

### Model Performance

| Metric | Brain Cancer Model | Breast Cancer Model |
|--------|-------------------|---------------------|
| **Architecture** | ResNet50 (3 classes) | ResNet50 (2 classes) |
| **Classes** | Glioma, Meningioma, Pituitary | Benign, Malignant |
| **Input Size** | 224 × 224 × 3 | 224 × 224 × 3 |
| **Inference Time** | < 100ms | < 100ms |
| **Device Support** | CPU, CUDA, MPS | CPU, CUDA, MPS |

### System Capabilities

- ✅ **Automatic Image Type Detection**: Correctly identifies Brain MRI vs Breast Histopathology
- ✅ **Real-time Inference**: Sub-second classification results
- ✅ **Probability Distribution**: Full class probability scores displayed
- ✅ **Severity Assessment**: Low/Medium/High risk classification
- ✅ **Clinical Recommendations**: Context-aware medical guidance
- ✅ **Responsive Web Interface**: Works across desktop and mobile devices
- ✅ **Docker Deployment**: Production-ready containerized architecture

### Output Features

1. **Primary Diagnosis**: Predicted cancer type with confidence percentage
2. **Probability Distribution**: Bar chart visualization of all class probabilities
3. **Severity Level**: Color-coded risk assessment (Low/Medium/High)
4. **Clinical Recommendations**: Specialist referral and follow-up testing suggestions
5. **Detailed Reports**: AI-generated comprehensive diagnostic reports

---

## 🎓 CONCLUSION

**OncoAI Fusion** successfully demonstrates the application of deep learning in multi-cancer diagnostic imaging. Key achievements include:

1. **Unified Multi-Cancer Platform**: Successfully integrated brain and breast cancer classification into a single, cohesive system with automatic image type detection.

2. **Transfer Learning Effectiveness**: ResNet50 pre-trained on ImageNet provides robust feature extraction, enabling high accuracy with relatively smaller medical imaging datasets.

3. **Clinical Applicability**: The system provides not just classification but actionable clinical insights including severity assessment and specialist recommendations.

4. **Modern Architecture**: The microservices-based design (React + FastAPI + PyTorch) ensures scalability, maintainability, and easy integration with existing healthcare systems.

5. **User Experience**: The intuitive drag-and-drop interface with real-time visualization makes the system accessible to medical professionals without technical expertise.

### Future Enhancements

- Expand to additional cancer types (Cervical, Kidney, Lung, Colon, Lymphoma, Oral)
- Integrate LLM-powered natural language report generation
- Implement explainable AI (Grad-CAM) for visualization of model attention
- Add multi-language support for global deployment
- Develop mobile applications for point-of-care diagnostics

---

## 📚 TECHNOLOGY STACK

| Component | Technology |
|-----------|------------|
| **Deep Learning** | PyTorch, TorchVision |
| **Model** | ResNet50 (Transfer Learning) |
| **Backend** | FastAPI (Python) |
| **Frontend** | React, Vite, Lucide Icons |
| **Deployment** | Docker, Docker Compose |
| **Hardware Support** | CPU, NVIDIA CUDA, Apple MPS |

---

> **Disclaimer**: This AI tool is for research and educational purposes only. It is not intended for clinical diagnosis and should not replace professional medical judgment.
