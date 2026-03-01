# 🫁 TB-Guard-XAI: Explainable AI Triage for Mass Tuberculosis Screening

**Mistral AI Worldwide Hackathon 2026 Submission**

TB-Guard-XAI is an Explainable AI triage and clinical decision-support pipeline designed to automate mass Tuberculosis (TB) screening in low-resource, high-burden environments where trained radiologists are scarce. 

Instead of acting as a "black box" that outputs a simple percentage, TB-Guard orchestrates a multi-modal ensemble of **Mistral's Latest AI Models** and **PyTorch Deep Learning** to provide robust, explainable, and localized clinical screening.

![TB-Guard-XAI Dashboard]([demo_dashboard_placeholder.png](https://github.com/vignesh19032005/TB-Guard-XAI/blob/de74fe2548342ca0c66d0f3771885d07c112c042/TB-Guard-XAI.png)) <!-- *Replace with actual screenshot of your beautiful UI before submitting!* -->

---

## 🚀 The Problem & The Mission
Tuberculosis kills 1.3 million people annually, with 87% of cases occurring in low-resource settings. The WHO explicitly endorses AI-assisted Chest X-Ray (CXR) screening to bridge the massive gap in healthcare personnel. However, existing AI models are often "black boxes" that clinicians cannot trust. 

**Our mission:** Build an AI system that knows *why* it made a decision, knows *when* it is uncertain, and uses Mistral's advanced reasoning to explain its thoughts just like a human doctor would.

---

## 🧠 Mistral AI Integration (The Core Stack)

TB-Guard-XAI is deeply integrated into the Mistral ecosystem, utilizing distinct models for specialized clinical agents:

1. **Mistral Vision (mistral-large-latest)**: Acts as the "Second Opinion Expert." It physically looks at the raw patient X-Ray, compares its findings against the PyTorch Deep Learning coordinates, and verifies abnormalities (Cavities, Opacities).
2. **Mistral Audio (voxtral-mini-latest)**: Powers the chaotic-environment input. Technicians can dictate symptoms via voice recording, and Voxtral instantly transcribes the clinical context without a keyboard.
3. **Mistral Triage Agent (mistral-small-latest)**: An ultra-fast safety validator that intercepts Voxtral voice input and instantly rejects queries/symptoms that are unrelated to respiratory illnesses.
4. **Mistral RAG Reasoner (mistral-large-latest)**: Utilizes native tool-calling to query a **Qdrant Vector Database** containing indexed WHO TB Guidelines, generating an evidence-backed, structured Clinical Explainer Report.
5. **MedGemma Safety Guardrails**: Validates the end chatbot responses to ensure clinical safety compliance.

---

## 🔬 Mathematical Architecture (Beyond the LLM)

To ensure true clinical safety, we built a bespoke Machine Learning pipeline beneath Mistral:
* **PyTorch CNN Ensemble:** Combines DenseNet121, EfficientNet-B4, and ResNet50 trained across 6 global datasets (Shenzhen, Montgomery, etc.).
* **Monte Carlo Dropout (Bayesian Uncertainty):** The neural network runs predictions 20 times per image. If it hallucinates or guesses, the mathematical variance flags the result as **"Unreliable — Review Required"**, overriding the base percentage.
* **Grad-CAM Heatmaps:** Generates a topological color map precisely showing the physician exactly *where* the AI found the infection in the lung.

---

## 🛠️ How to Run Locally

### 1. Requirements
Ensure you have Python 3.10+ installed.
```bash
git clone https://github.com/your-username/TB-Guard-XAI.git
cd TB-Guard-XAI
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables
You need your Mistral API key to run the clinical reasoning layers. Create a `.env` file in the root directory:
```env
MISTRAL_API_KEY=your_mistral_key_here
```

### 3. Run the Server
```bash
python backend.py
```
*Open your browser to `http://127.0.0.1:8000` to access the Clinical Dashboard.*

---

## 📑 Clinical Disclaimer
**Not for self-diagnosis.** TB-Guard-XAI is an experimental clinical decision-support tool built for hackathon demonstration. It is designed to assist trained medical technicians as a primary triage filter. All positive results must be confirmed via Sputum Xpert MTB/RIF or culture tests in accordance with WHO guidelines.

> Built with ❤️ for the Mistral AI Hackathon 2026.
