---
title: TB Guard XAI
emoji: 🏥
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: backend.py
pinned: true
short_description: TB Clinical Triage with Deep Learning, RAG & Explainable AI
---
# 🫁 TB-Guard-XAI: Explainable AI Triage for Mass Tuberculosis Screening

**Built for the Mistral AI Worldwide Hackathon 2026**

> TB-Guard-XAI is an explainable, multimodal clinical triage engine. Uniting PyTorch deep learning with Mistral, Bayesian Uncertainty mathematically detects AI "guessing," while Grad-CAM++ heatmaps highlight infections. Mistral Vision adds a 2nd opinion, Voxtral transcribes voice, and RAG outputs MedGemma-safe, WHO-backed clinical reports.

[![Hugging Face Space](https://img.shields.io/badge/🤗_Space-Live_Demo-blue)](https://huggingface.co/spaces/mistral-hackaton-2026/TB-Guard-XAI)
[![Demo Video](https://img.shields.io/badge/🎬_Video-Watch_Pitch-red)](https://youtu.be/UyxZCp2q7TM)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![TB-Guard-XAI Dashboard](https://github.com/vignesh19032005/TB-Guard-XAI/blob/de74fe2548342ca0c66d0f3771885d07c112c042/TB-Guard-XAI.png)

---

## 🚀 The Clinical Problem
Tuberculosis kills 1.3 million people annually, with 87% of cases occurring in low-resource settings. The WHO explicitly endorses AI-assisted Chest X-Ray (CXR) screening to bridge the massive gap in healthcare personnel. 

**The Flaw in Current AI:** Existing medical AI models are *"black boxes"*. They output a rigid probability (e.g., "95% TB") using standard softmax functions. This results in **false overconfidence**. If given an obscure anomaly, traditional AI will confidently hallucinate a diagnosis because it lacks the mathematical capacity to say, *"I don't know."* Furthermore, they provide no explanation for *why* they made the decision, making them unsafe for autonomous triage.

**Our Mission:** Build an AI system that knows *why* it made a decision, mathematically calculates *when* it is out of its depth, and orchestrates the Mistral AI ecosystem to explain its reasoning exactly as a human doctor would.

---

## 🧠 The Architecture & Tech Stack Justification
TB-Guard-XAI is not a simple wrapper around an LLM. It is a highly engineered, multi-agent pipeline bridging deterministic Deep Learning with non-deterministic Generative AI.

### 1. The Mistral AI Ecosystem (The Brains)
We utilized almost the entire suite of Mistral's latest models, assigning them specialized agentic roles:

* **👁️ Mistral Vision (`mistral-large-latest`): The Second Opinion.**
  * *Why this?* Instead of relying solely on our PyTorch CNN, we pass the compressed X-Ray directly to Mistral Large. It acts as an independent radiologist, cross-verifying the mathematical coordinates found by PyTorch and hunting for contextual clues like Lymphadenopathy or Cavitations.
* **🎙️ Voxtral Audio (`voxtral-mini-latest`): Acoustic Context.**
  * *Why this?* Rural clinics are chaotic. Technicians don't have time to type. Voxtral ingests spoken symptoms ("Patient has night sweats") and transcribes them instantly.
* **🛡️ Mistral Router (`mistral-small-latest`): The Safety Gatekeeper.**
  * *Why this?* We use Mistral Small for zero-latency, ultra-cheap intent classification. It intercepts the transcribed voice notes. If a patient describes a broken ankle, Mistral Small instantly blocks the query for violating the Respiratory domain, preserving clinical compliance.
* **📚 Mistral RAG Reasoner (`mistral-large-latest`): Clinical Synthesis.**
  * *Why this?* Mistral Large possesses exceptional native tool-calling. It dynamically queries our Qdrant Vector Database (loaded with WHO TB Guidelines) and fuses the RAG evidence, Mistral Vision's visual assessment, and PyTorch's mathematical probabilities into a cohesive, structured Medical Report.
* **⚖️ MedGemma: End-of-Line Validation.**
  * *Why this?* Used as a secondary open-weight safety validator to ensure the final generated advice does not provide definitive medical diagnoses, keeping the tool strictly as "Decision Support."

### 2. The Deep Learning Engine (The Eyes & The Math)
Beneath the LLMs lies a robust computer vision pipeline designed for maximum explainability.

* **Convoluted Neural Network (CNN) Ensemble**
  * *What is it?* A parallel architecture fusing DenseNet121, EfficientNet-B4, and ResNet50.
  * *Why average them?* Single models inherit inherent dataset biases. By ensembling three distinct architectures, we eliminate distinct blind spots. Furthermore, they are trained on 6 distinct global CXR datasets (Shenzhen, Montgomery, etc.) to ensure ethnic and anatomical generalization.
* **Bayesian Deep Learning: Monte Carlo (MC) Dropout**
  * *What is it?* The crown jewel of our safety mechanism. Standard AI evaluates an image once. MC Dropout forces our neural network to evaluate the same X-Ray **20 different times**, randomly turning off ("dropping out") different neurons during each pass.
  * *Why use it?* If the model is recognizing true TB features, the 20 predictions will be nearly identical (Low Variance). But if the model is guessing on an anomalous image, the 20 predictions will wildly disagree (High Variance). When high variance is detected, the system overrides the probability and flags **"Unreliable — Human Review Required,"** legally protecting the clinic from false AI confidence.
* **Explainable AI: Grad-CAM++ (Gradient-weighted Class Activation Mapping)**
  * *What is it?* An algorithm that traces the classification logic backwards through the CNN to find exactly which pixels activated the "Tuberculosis" neurons.
  * *Why use it?* It generates a topological heatmap over the X-Ray. Doctors don't have to trust the AI blindly; they can physically see exactly what the AI is looking at. 

### 3. The Infrastructure Pipeline
* **FastAPI (Backend):** Chosen over Flask/Django for its asynchronous performance capability, crucial for handling concurrent PyTorch inference, Mistral tool-calling, and Audio processing simultaneously.
* **Qdrant (Vector Database):** Chosen over Pinecone/Milvus for its incredible local-deployment capability and dense vector search speeds, serving our WHO RAG context instantly.
* **Vanilla HTML/JS + Tailwind (Frontend):** We specifically avoided heavy React/Next.js frameworks to guarantee the UI could run on extremely low-end, low-RAM hospital registry computers with zero dependency bloat.

---

## 💡 Key Features at a Glance
* **Drag-and-Drop X-Ray Analysis** with Native Bayesian Uncertainty bounds.
* **Mistral Vision Multimodal Verification** natively embedded in the UI.
* **Voice-Activated Clinical Context** powered by Voxtral.
* **Grad-CAM++ Topological Visualizations.**
* **Built-in AI Respiratory Chatbot.**
* **One-Click Printable PDF Triage Reports** for lab handover.

---

## 🌐 Live Deployment
TB-Guard-XAI is packaged and deployed on **Hugging Face Spaces**. You can run the live demo, upload X-rays, record voice notes, and test clinical queries directly via the cloud.

🔗 **[Launch TB-Guard-XAI on Hugging Face](https://huggingface.co/spaces/mistral-hackaton-2026/TB-Guard-XAI)**

---

## 🛠️ Run It Locally

### 1. Setup & Install
Ensure you have Python 3.10+ installed.
```bash
git clone https://github.com/vignesh19032005/TB-Guard-XAI.git
cd TB-Guard-XAI
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables
You need your Mistral API key to run the active inference pipelines. Create a `.env` file in the root directory:
```env
MISTRAL_API_KEY=your_mistral_key_here
```

### 3. Run the Local FastApi Server
```bash
python backend.py
```
*Open your browser to `http://127.0.0.1:8000` to access the full UI.*

---

## 🔮 What's Next for TB-Guard-XAI?
- **Automated PACS Watcher:** We are actively building an offline background folder-watcher to automatically ingest and triage batch X-Rays dumped from hospital local drives.
- **Continuous Learning Loop:** Implementing human-in-the-loop validation where physicians can correct Mistral via the UI, feeding the verified data back into the underlying ensemble.
- **DICOM Support:** Transitioning from PNG parsing to native HL7/DICOM medical file support for true hospital system interoperability.

---

### 📑 Clinical Disclaimer
**Not for self-diagnosis.** TB-Guard-XAI is an experimental clinical decision-support tool built specifically for the **Mistral AI Worldwide Hackathon 2026** demonstration. It is designed to assist trained medical technicians as a primary triage filter. All positive and unsure results must lead to confirmatory Sputum Xpert MTB/RIF or culture tests in accordance with local WHO guidelines.

> *Built with ❤️ for Mistral AI. Code by Vignesh.*
