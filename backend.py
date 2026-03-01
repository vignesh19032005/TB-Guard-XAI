# backend.py
# Phase 14 Arch: FastAPI Backend Service

import os
import torch
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import shutil
from pathlib import Path

# Disable deprecation warning for st.image
import warnings
warnings.filterwarnings("ignore", message=".*use_column_width parameter has been deprecated.*")

try:
    from mistral_explainer import MistralExplainer
except Exception as e:
    print(f"⚠️ Failed to import MistralExplainer: {e}")
    import traceback
    traceback.print_exc()
    MistralExplainer = None

app = FastAPI(title="TB-Guard-XAI Clinical API", description="AI Screening for Tuberculosis", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
explainer = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("startup")
async def startup_event():
    global explainer
    print("🚀 Initializing Clinical AI Models...")
    if MistralExplainer is None:
        print("❌ MistralExplainer not available (import failed). Check errors above.")
        return
    try:
        explainer = MistralExplainer(model_path="models/ensemble_best.pth")
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not load models during startup: {e}")

class ClinicalRequest(BaseModel):
    symptoms: str = ""

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/consult", response_class=HTMLResponse)
async def consult(request: Request):
    return templates.TemplateResponse("consult.html", {"request": request})

@app.get("/gallery", response_class=HTMLResponse)
async def gallery(request: Request):
    return templates.TemplateResponse("gallery.html", {"request": request})

@app.get("/status")
def status():
    return {"status": "online", "model_device": DEVICE, "rag_ready": True}

@app.post("/analyze")
async def analyze_xray(
    file: UploadFile = File(...),
    symptoms: str = Form(""),
    threshold: float = Form(0.42),
    age_group: str = Form("Adult (40-64)")
):
    """Analyze X-ray and return prediction + explanation from ensemble + LLM"""
    if explainer is None:
        return {"error": "Model failed to load"}

    # Save temp file
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / file.filename
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Run deep inference pipeline (Phase 1/2 + Phase 4)
        result = explainer.explain(str(temp_path), symptoms=symptoms, threshold=threshold, age_group=age_group)
        
        # Cleanup
        os.remove(temp_path)
        
        return {
            "prediction": result["prediction"],
            "probability": float(result["probability"]),
            "uncertainty": result["uncertainty"],
            "uncertainty_std": float(result["uncertainty_std"]),
            "region": result.get("gradcam_region", "Lung Field"),
            "clinical_explanation": result["explanation"],
            "evidence": result.get("evidence", []),
            "gradcam_image": result.get("gradcam_image"),
            "gradcam_available": result.get("gradcam_image") is not None
        }
        
    except Exception as e:
        if temp_path.exists():
            os.remove(temp_path)
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio recording of patient symptoms using Voxtral"""
    if explainer is None:
        return {"error": "Model not loaded"}
    
    try:
        audio_bytes = await file.read()
        transcript = explainer.transcribe_audio(audio_bytes)
        
        if transcript:
            is_valid = explainer.validate_symptoms(transcript)
            if not is_valid:
                return {
                    "error": "WARNING: These symptoms do not appear to be related to respiratory, chest, or tuberculosis conditions.", 
                    "transcript": transcript,
                    "is_valid": False
                }
            return {"transcript": transcript, "is_valid": True}
        else:
            return {"error": "Transcription failed", "transcript": "", "is_valid": False}
    except Exception as e:
        return {"error": str(e), "transcript": ""}

class ConsultRequest(BaseModel):
    query: str

@app.post("/general_consult")
async def general_consult(request: ConsultRequest):
    """General Medical Consult using Mistral and MedGemma Validation"""
    if explainer is None or explainer.mistral is None:
        return {"response": "Mistral API not configured", "safety_validated": False}
    
    query = request.query
    
    # 1. Mistral Large 3 generates differential diagnosis
    messages = [
        {"role": "system", "content": "You are a specialized Respiratory & TB clinical decision support AI. Provide concise, structured differential diagnoses ONLY for respiratory and chest-related conditions. If the user asks about non-respiratory medical topics (e.g. broken bones, dermatology, general abdominal pain), politely decline by stating you are specifically trained for Pulmonary and Tuberculosis clinical support."},
        {"role": "user", "content": query}
    ]
    
    try:
        response = explainer.mistral.chat.complete(
            model="mistral-large-latest",
            messages=messages,
            temperature=0.2,
            max_tokens=4000
        )
        mistral_response = response.choices[0].message.content
        
        # 2. MedGemma Validator (Simulation / Basic implementation using Mistral as proxy if MedGemma not available)
        # Real-world we'd call a local MedGemma model. Here we use Mistral with a different prompt to validate.
        validator_messages = [
            {"role": "system", "content": "You are MedGemma, a strict clinical safety validator. Assess the proposed medical advice for critical safety risks. Return EXACTLY 'SAFE' or 'UNSAFE' with no other text."},
            {"role": "user", "content": f"Query: {query}\n\nSuggested Advice: {mistral_response}\n\nIs this advice clinically safe?"}
        ]
        
        val_response = explainer.mistral.chat.complete(
            model="mistral-large-latest",
            messages=validator_messages,
            temperature=0.0,
            max_tokens=10
        )
        val_text = val_response.choices[0].message.content.strip().upper()
        is_safe = "SAFE" in val_text
        
        return {
            "response": mistral_response,
            "safety_validated": is_safe
        }
    except Exception as e:
        return {"error": str(e), "response": "Error processing request.", "safety_validated": False}

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
