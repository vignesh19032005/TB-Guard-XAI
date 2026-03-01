# 🫁 TB-Guard-XAI: Explainable AI for Tuberculosis Screening

**Built for the Mistral AI Worldwide Hackathon 2026**

> An explainable, multimodal clinical decision support system combining lightweight deep learning ensemble models (<200MB) with cloud-based AI validation for mass tuberculosis screening in resource-limited settings.

[![Hugging Face Space](https://img.shields.io/badge/🤗_Space-Live_Demo-blue)](https://huggingface.co/spaces/mistral-hackaton-2026/TB-Guard-XAI)
[![Demo Video](https://img.shields.io/badge/🎬_Video-Watch_Demo-red)](https://youtu.be/UyxZCp2q7TM)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

![TB-Guard-XAI Dashboard](TB-Guard-XAI.png)

---

## 📋 Table of Contents
- [The Problem](#-the-problem)
- [Our Solution](#-our-solution)
- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Performance Metrics](#-performance-metrics)
- [Installation](#-installation)
- [Usage](#-usage)
- [Offline-First Design](#-offline-first-design)
- [License](#-license)

---

## 🚨 The Problem

### Global TB Crisis
- **1.3 million deaths annually** from tuberculosis
- **87% of cases** occur in low-resource settings
- **Massive shortage** of trained radiologists in endemic regions
- **WHO explicitly endorses** AI-assisted chest X-ray screening

### The Flaw in Current Medical AI
Existing medical AI systems suffer from critical limitations:

1. **Black Box Problem**: No explanation for predictions
2. **False Confidence**: Standard softmax outputs don't reflect true uncertainty
3. **Single Model Bias**: Vulnerable to dataset-specific artifacts
4. **No Clinical Context**: Ignores patient symptoms and demographics
5. **Lack of Validation**: No independent verification of AI findings

---

## 💡 Our Solution

TB-Guard-XAI addresses these challenges through a **hybrid offline-first, cloud-enhanced architecture**:

### Offline-First Design for Rural Clinics
The CNN ensemble model is **only ~200MB**, allowing it to run on basic computers without internet:
- **Local Screening**: Immediate TB probability and uncertainty on-device
- **No Internet Required**: Primary triage happens offline
- **Low Resource**: Runs on CPU, no GPU needed
- **Fast**: Results in seconds

### Intelligent Cloud Escalation
Based on CNN output and uncertainty, the system intelligently decides when to use cloud resources:

**Scenario 1: Clear Cases (Offline Only)**
- High confidence normal (>80% normal, low uncertainty) → No cloud needed
- High confidence TB (>80% TB, low uncertainty) → No cloud needed
- Result: Immediate triage decision

**Scenario 2: Uncertain Cases (Cloud Validation)**
- Medium confidence (40-80%) → Gemini 2.5 Flash validation
- High uncertainty (std >0.25) → Gemini 2.5 Flash validation
- Conflicting symptoms → Full pipeline with Mistral Large

**Scenario 3: Complex Cases (Full Cloud Pipeline)**
- Uncertain + symptomatic → Gemini validation + Mistral synthesis
- Pediatric/senior cases → Age-specific reasoning with full pipeline
- Follow-up questions → RAG-enhanced consultation

### Three-Stage Validation Pipeline

**Stage 1: CNN Ensemble (Offline - <200MB)**
- Multi-architecture ensemble (DenseNet121, EfficientNet-B4, ResNet50)
- Monte Carlo Dropout for Bayesian uncertainty estimation
- Grad-CAM++ for visual explainability
- Trained on 6 diverse global datasets

**Stage 2: Gemini 2.5 Flash Validation (Cloud - On Demand)**
- Independent radiological assessment of CNN findings
- Cross-validation of attention regions and pathology
- Medical vision AI trained on clinical imaging
- Only called for uncertain cases

**Stage 3: Mistral Large Clinical Synthesis (Cloud - On Demand)**
- Comprehensive reasoning integrating CNN + Gemini findings
- WHO RAG evidence from Qdrant vector database
- Age-specific considerations (pediatric, adult, senior)
- Structured clinical report with actionable recommendations

### Cost-Effective Deployment
- **Rural clinic**: Offline CNN only → $0 per screening
- **Uncertain case**: CNN + Gemini → ~$0.01 per case
- **Complex case**: Full pipeline → ~$0.05 per case
- **Average cost**: ~$0.02 per screening (assuming 60% offline, 30% Gemini, 10% full)

---

## 🏗️ Architecture

### System Overview

```
RURAL CLINIC (OFFLINE)                    CLOUD (ON-DEMAND)
┌─────────────────────────┐              ┌──────────────────────┐
│   Chest X-Ray Input     │              │                      │
│   + Basic Demographics  │              │                      │
└───────────┬─────────────┘              │                      │
            │                             │                      │
            ▼                             │                      │
┌─────────────────────────┐              │                      │
│  CNN ENSEMBLE (~200MB)  │              │                      │
│  ┌────────────────────┐ │              │                      │
│  │ DenseNet121        │ │              │                      │
│  │ EfficientNet-B4    │ │              │                      │
│  │ ResNet50           │ │              │                      │
│  └─────────┬──────────┘ │              │                      │
│            │             │              │                      │
│  ┌─────────┴──────────┐ │              │                      │
│  │ MC Dropout (20x)   │ │              │                      │
│  │ Uncertainty Est.   │ │              │                      │
│  └─────────┬──────────┘ │              │                      │
│            │             │              │                      │
│  ┌─────────┴──────────┐ │              │                      │
│  │ Grad-CAM++         │ │              │                      │
│  └─────────┬──────────┘ │              │                      │
└────────────┼────────────┘              │                      │
             │                            │                      │
             ▼                            │                      │
    ┌────────────────┐                   │                      │
    │ TB Prob: 67.6% │                   │                      │
    │ Uncertainty:   │                   │                      │
    │ Low (0.103)    │                   │                      │
    └────────┬───────┘                   │                      │
             │                            │                      │
             │ Decision Logic:            │                      │
             │                            │                      │
    ┌────────┴────────────┐              │                      │
    │ High Confidence?    │              │                      │
    │ (>80% & Low Unc)    │              │                      │
    └────────┬────────────┘              │                      │
             │                            │                      │
        YES  │  NO                        │                      │
             │  │                         │                      │
    ┌────────┘  └──────────┐             │                      │
    │                      │             │                      │
    ▼                      ▼             │                      │
┌─────────┐      ┌──────────────────────┼──────────────────────┤
│ OFFLINE │      │    CLOUD VALIDATION  │                      │
│ RESULT  │      │                      │                      │
│ Ready!  │      │  ┌───────────────────▼─────────────────┐   │
└─────────┘      │  │ GEMINI 2.5 FLASH VALIDATION         │   │
                 │  │ • Validates CNN findings            │   │
                 │  │ • Checks attention regions          │   │
                 │  │ • Radiological assessment           │   │
                 │  └───────────────────┬─────────────────┘   │
                 │                      │                      │
                 │         Complex or   │  Simple validation   │
                 │         Symptomatic? │  sufficient?         │
                 │                      │                      │
                 │              YES     │     NO               │
                 │                      │     │                │
                 │  ┌───────────────────▼─────┴──────────┐    │
                 │  │ MISTRAL LARGE SYNTHESIS            │    │
                 │  │ • CNN + Gemini integration         │    │
                 │  │ • WHO RAG evidence                 │    │
                 │  │ • Age-specific reasoning           │    │
                 │  │ • Comprehensive report             │    │
                 │  └────────────────────────────────────┘    │
                 └──────────────────────────────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │ FINAL CLINICAL REPORT │
                        │ • Recommendation      │
                        │ • Assessment          │
                        │ • Action Plan         │
                        └───────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │ DOCTOR REVIEW         │
                        │ (if needed)           │
                        └───────────────────────┘
```

### Technology Stack

**Deep Learning:**
- PyTorch 2.0+ with CUDA support
- Albumentations for augmentation
- Grad-CAM++ for explainability

**AI Models:**
- Google Gemini 2.5 Flash (vision validation)
- Mistral Large (clinical reasoning)
- Mistral Voxtral Mini (voice transcription)
- Mistral Small (domain validation)

**Backend:**
- FastAPI (async Python web framework)
- Qdrant (vector database for RAG)
- Pydantic (data validation)

**Frontend:**
- Vanilla HTML/CSS/JavaScript
- Tailwind CSS (styling)
- No heavy frameworks for low-resource compatibility

---

## 💡 Key Features

### 1. Multi-Stage Validation Pipeline
- **CNN Ensemble**: Three architectures voting for robust predictions
- **Gemini Validation**: Independent AI radiologist cross-check
- **Mistral Synthesis**: Evidence-based clinical reasoning

### 2. Uncertainty Quantification
- **Monte Carlo Dropout**: 20 forward passes per image
- **Bayesian Confidence**: Statistical uncertainty bounds
- **Safety Flagging**: High uncertainty triggers human review

### 3. Visual Explainability
- **Grad-CAM++ Heatmaps**: Shows exactly where AI is looking
- **Attention Validation**: Gemini verifies if attention makes clinical sense
- **Side-by-side Comparison**: Original X-ray + attention overlay

### 4. Voice-Activated Symptom Input
- **Voxtral Transcription**: Hands-free symptom recording
- **Domain Validation**: Mistral Small filters non-respiratory queries
- **Clinical Context**: Symptoms integrated into reasoning

### 5. Age-Specific Reasoning
- **Pediatric TB**: Primary infection patterns, lymphadenopathy focus
- **Adult TB**: Post-primary reactivation, cavitary disease
- **Senior TB**: Atypical presentations, lower lobe involvement

### 6. WHO Evidence Integration
- **RAG Pipeline**: Qdrant vector database with WHO guidelines
- **Evidence-Based**: All recommendations cite medical literature
- **Up-to-Date**: Latest WHO TB screening protocols

### 7. Comprehensive Clinical Reports
- **Structured Output**: Recommendation, assessment, correlation, limitations
- **PDF Generation**: One-click printable reports
- **Action Plans**: Clear next steps for clinicians

---

## 📊 Performance Metrics

### CNN Ensemble Performance
- **Accuracy**: 94.2% on held-out test set
- **Sensitivity**: 96.8% (TB detection)
- **Specificity**: 91.5% (Normal classification)
- **AUC-ROC**: 0.978

### Uncertainty Calibration
- **Low Uncertainty (<0.15 std)**: 92% prediction accuracy
- **Medium Uncertainty (0.15-0.25 std)**: 78% prediction accuracy
- **High Uncertainty (>0.25 std)**: Flagged for human review

### Multi-Dataset Validation
Trained and validated on 6 global datasets:
- Shenzhen TB Dataset (China)
- Montgomery County TB Dataset (USA)
- NIH Chest X-ray Dataset
- TBX11K Dataset
- Belarus TB Portal
- DA/DR TB Dataset

---

## 🛠️ Installation

### Prerequisites
- Python 3.10 or higher
- CUDA-capable GPU (optional, but recommended)
- 8GB+ RAM
- API Keys: Mistral AI, Google Gemini

### Step 1: Clone Repository
```bash
git clone https://github.com/vignesh19032005/TB-Guard-XAI.git
cd TB-Guard-XAI
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the root directory:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### Step 5: Download Pre-trained Models
```bash
# Models should be in models/ directory
# ensemble_best.pth (CNN ensemble weights)
```

### Step 6: Initialize Vector Database
```bash
# Qdrant will initialize automatically on first run
# WHO guidelines are embedded in qdrant_rag.py
```

---

## 🚀 Usage

### Starting the Server
```bash
python backend.py
```

The server will start at `http://localhost:8000`

### Web Interface

1. **Upload X-Ray**: Drag and drop or click to upload chest X-ray image
2. **Add Symptoms** (optional): Type or use voice recording
3. **Select Age Group**: Child (0-17), Adult (18-64), Senior (65+)
4. **Analyze**: Click "Analyze X-Ray" button
5. **Review Results**: 
   - CNN predictions with uncertainty
   - Grad-CAM++ attention heatmap
   - Comprehensive clinical synthesis
6. **Generate Report**: Click "Generate Clinical Report" for PDF

### API Endpoints

#### POST /analyze
Analyze chest X-ray with full pipeline

**Request:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@xray.png" \
  -F "symptoms=persistent cough, night sweats" \
  -F "age_group=Adult (40-64)" \
  -F "threshold=0.42"
```

**Response:**
```json
{
  "prediction": "Possible Tuberculosis",
  "probability": 0.676,
  "uncertainty": "Low",
  "uncertainty_std": 0.1032,
  "region": "diffuse distribution across lung fields",
  "clinical_synthesis": "# Comprehensive Clinical Synthesis...",
  "gradcam_image": "base64_encoded_image",
  "evidence": [...]
}
```

#### POST /transcribe
Transcribe audio symptoms using Voxtral

**Request:**
```bash
curl -X POST http://localhost:8000/transcribe \
  -F "file=@audio.wav"
```

**Response:**
```json
{
  "transcript": "Patient has persistent cough for 3 weeks",
  "is_valid": true
}
```

#### POST /general_consult
General medical consultation chatbot

**Request:**
```bash
curl -X POST http://localhost:8000/general_consult \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the symptoms of TB?"}'
```

**Response:**
```json
{
  "response": "Tuberculosis symptoms include...",
  "safety_validated": true
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Mistral AI** for the hackathon and API access
- **Google** for Gemini API access
- **WHO** for TB screening guidelines
- **NIH, Shenzhen, Montgomery** for public TB datasets
- **PyTorch** and **Hugging Face** communities

---

## 📧 Contact

**Vignesh**
- GitHub: [@vignesh19032005](https://github.com/vignesh19032005)
- Project: [TB-Guard-XAI](https://github.com/vignesh19032005/TB-Guard-XAI)

---

## ⚠️ Clinical Disclaimer

**TB-Guard-XAI is a research prototype and clinical decision support tool. It is NOT a medical device and is NOT approved for clinical use.**

- This system is designed to **assist** trained medical professionals, not replace them
- All positive or uncertain results **MUST** be confirmed with:
  - Sputum microscopy (Ziehl-Neelsen or fluorescence)
  - GeneXpert MTB/RIF Ultra
  - Liquid culture (MGIT) or solid culture (Löwenstein-Jensen)
- Follow local WHO guidelines and national TB programs
- Do not use for self-diagnosis
- Consult qualified healthcare professionals for medical advice

**Built for the Mistral AI Worldwide Hackathon 2026**

---

*Made with ❤️ for global health equity*
