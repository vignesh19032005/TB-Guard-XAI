# backend.py
# Phase 14 Arch: FastAPI Backend Service

import os
import torch
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import shutil
from pathlib import Path
import zipfile
import io
import json
from datetime import datetime
from typing import List

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
        print(f"❌ ERROR: Could not load models: {e}")
        import traceback
        traceback.print_exc()

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
    return {
        "status": "online" if explainer else "error",
        "model_device": DEVICE,
        "rag_ready": True
    }

@app.get("/health")
def health():
    """Simple health check"""
    return {"status": "ok"} if explainer else {"status": "error"}

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

    # Simple validation
    # 1. Check file extension
    allowed_ext = ['.jpg', '.jpeg', '.png', '.webp']
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_ext:
        return {"error": f"Invalid file type. Allowed: {', '.join(allowed_ext)}"}
    
    # 2. Check file size (max 50MB)
    contents = await file.read()
    if len(contents) > 50 * 1024 * 1024:
        return {"error": "File too large. Maximum size: 50MB"}
    if len(contents) == 0:
        return {"error": "Empty file"}
    
    # 3. Validate it's actually an image
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(contents))
        img.verify()
        # Check dimensions
        img = Image.open(io.BytesIO(contents))
        w, h = img.size
        if w < 100 or h < 100 or w > 10000 or h > 10000:
            return {"error": f"Invalid image dimensions: {w}x{h}"}
    except Exception as e:
        return {"error": f"Invalid image file: {str(e)}"}

    # Save temp file
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    import uuid
    temp_path = temp_dir / f"{uuid.uuid4().hex}{file_ext}"
    
    # Write file
    try:
        with open(temp_path, "wb") as buffer:
            buffer.write(contents)
    except Exception as e:
        return {"error": f"Failed to save file: {str(e)}"}
        
    try:
        # Run deep inference pipeline (Phase 1/2 + Phase 4)
        result = explainer.explain(str(temp_path), symptoms=symptoms, threshold=threshold, age_group=age_group)
        
        # Cleanup
        try:
            os.remove(temp_path)
        except:
            pass
        
        # explanation is now a single string
        explanation = result["explanation"]
        print(f"📋 Explanation type: {type(explanation)}, length: {len(explanation) if isinstance(explanation, str) else 'N/A'}")
        
        return {
            "prediction": result["prediction"],
            "probability": float(result["probability"]),
            "uncertainty": result["uncertainty"],
            "uncertainty_std": float(result["uncertainty_std"]),
            "region": result.get("gradcam_region", "Lung Field"),
            "clinical_synthesis": explanation,
            "evidence": result.get("evidence", []),
            "gradcam_image": result.get("gradcam_image"),
            "gradcam_available": result.get("gradcam_image") is not None,
            "mode": result.get("mode", "unknown")  # offline or online
        }
        
    except Exception as e:
        # Cleanup on error
        try:
            os.remove(temp_path)
        except:
            pass
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribe audio recording of patient symptoms using Voxtral"""
    if explainer is None:
        return {"error": "Model not loaded", "transcript": "", "is_valid": False}
    
    # Simple validation
    # 1. Check file size (max 25MB)
    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        return {"error": "Empty audio file", "transcript": "", "is_valid": False}
    if len(audio_bytes) > 25 * 1024 * 1024:
        return {"error": "Audio file too large (max 25MB)", "transcript": "", "is_valid": False}
    
    # 2. Check file extension
    allowed_ext = ['.wav', '.mp3', '.m4a', '.ogg', '.webm']
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_ext:
        return {"error": f"Invalid audio format. Allowed: {', '.join(allowed_ext)}", "transcript": "", "is_valid": False}
    
    try:
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
    """General Medical Consult using Mistral Large with Enhanced Medical Prompting"""
    if explainer is None or explainer.mistral is None:
        return {"response": "Mistral API not configured", "safety_validated": True}
    
    query = request.query
    
    # Enhanced system prompt for medical accuracy and safety
    system_prompt = """You are a specialized Respiratory & TB clinical decision support AI with the following expertise:

CORE COMPETENCIES:
- Pulmonary medicine and tuberculosis diagnosis
- Chest radiology interpretation
- Differential diagnosis of respiratory conditions
- Evidence-based clinical guidelines (WHO, CDC)
- Age-specific TB presentations
- Drug interactions and treatment protocols

CLINICAL APPROACH:
- Provide structured, evidence-based responses
- Consider differential diagnoses systematically
- Reference clinical guidelines when applicable
- Acknowledge limitations and uncertainties
- Recommend appropriate follow-up and testing

SAFETY PROTOCOLS:
- Never provide definitive diagnoses (screening support only)
- Always recommend professional medical consultation
- Flag urgent/emergency symptoms immediately
- Decline non-respiratory medical topics politely
- Maintain clinical precision and accuracy

RESPONSE FORMAT:
- Use clear, professional medical terminology
- Structure responses logically (assessment → reasoning → recommendation)
- Cite evidence levels when possible
- Be concise yet comprehensive (2-4 paragraphs)

SCOPE LIMITATIONS:
If asked about non-respiratory topics (orthopedics, dermatology, general abdominal pain, etc.), politely state:
"I am specifically trained for Pulmonary and Tuberculosis clinical support. For [topic], please consult an appropriate specialist."
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    try:
        response = explainer.mistral.chat.complete(
            model="mistral-large-latest",
            messages=messages,
            temperature=0.15,  # Lower temperature for medical precision
            max_tokens=4000
        )
        mistral_response = response.choices[0].message.content
        
        # Simple safety check: flag if response contains concerning patterns
        safety_validated = True
        concerning_patterns = [
            "definitely have", "certainly have", "you have cancer",
            "no need to see a doctor", "don't consult", "ignore symptoms"
        ]
        
        response_lower = mistral_response.lower()
        for pattern in concerning_patterns:
            if pattern in response_lower:
                safety_validated = False
                break
        
        return {
            "response": mistral_response,
            "safety_validated": safety_validated
        }
    except Exception as e:
        return {"error": str(e), "response": "Error processing request.", "safety_validated": False}

@app.post("/batch_analyze_stream")
async def batch_analyze_stream(files: List[UploadFile] = File(...)):
    """Batch process with real-time progress updates via Server-Sent Events"""
    if explainer is None:
        return {"error": "Model failed to load"}
    
    if not files or len(files) == 0:
        return {"error": "No files uploaded"}
    
    if len(files) > 100:
        return {"error": "Maximum 100 files allowed per batch"}
    
    async def generate_progress():
        import asyncio
        
        # Create temp directories
        batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_temp = Path("batch_temp") / batch_id
        batch_temp.mkdir(parents=True, exist_ok=True)
        
        reports_dir = batch_temp / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        results = []
        processed_count = 0
        total_files = len(files)
        
        # Send initial progress
        yield f"data: {json.dumps({'type': 'start', 'total': total_files, 'batch_id': batch_id})}\n\n"
        await asyncio.sleep(0.1)
        
        for idx, file in enumerate(files):
            try:
                # Send progress update
                yield f"data: {json.dumps({'type': 'processing', 'current': idx + 1, 'total': total_files, 'filename': file.filename})}\n\n"
                await asyncio.sleep(0.1)
                
                # Validate file
                allowed_ext = ['.jpg', '.jpeg', '.png', '.webp']
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in allowed_ext:
                    results.append({
                        "filename": file.filename,
                        "status": "error",
                        "error": f"Invalid file type: {file_ext}"
                    })
                    yield f"data: {json.dumps({'type': 'error', 'current': idx + 1, 'filename': file.filename, 'error': f'Invalid file type: {file_ext}'})}\n\n"
                    await asyncio.sleep(0.1)
                    continue
                
                # Read and validate
                contents = await file.read()
                if len(contents) > 50 * 1024 * 1024:
                    results.append({
                        "filename": file.filename,
                        "status": "error",
                        "error": "File too large (max 50MB)"
                    })
                    yield f"data: {json.dumps({'type': 'error', 'current': idx + 1, 'filename': file.filename, 'error': 'File too large'})}\n\n"
                    await asyncio.sleep(0.1)
                    continue
                
                # Validate image
                from PIL import Image
                import io as iolib
                try:
                    img = Image.open(iolib.BytesIO(contents))
                    img.verify()
                    img = Image.open(iolib.BytesIO(contents))
                    w, h = img.size
                    if w < 100 or h < 100 or w > 10000 or h > 10000:
                        results.append({
                            "filename": file.filename,
                            "status": "error",
                            "error": f"Invalid dimensions: {w}x{h}"
                        })
                        yield f"data: {json.dumps({'type': 'error', 'current': idx + 1, 'filename': file.filename, 'error': f'Invalid dimensions'})}\n\n"
                        await asyncio.sleep(0.1)
                        continue
                except Exception as e:
                    results.append({
                        "filename": file.filename,
                        "status": "error",
                        "error": f"Invalid image: {str(e)}"
                    })
                    yield f"data: {json.dumps({'type': 'error', 'current': idx + 1, 'filename': file.filename, 'error': 'Invalid image'})}\n\n"
                    await asyncio.sleep(0.1)
                    continue
                
                # Save temp file
                temp_path = batch_temp / file.filename
                with open(temp_path, "wb") as buffer:
                    buffer.write(contents)
                
                # Convert image to base64 for PDF
                import base64
                original_image_base64 = base64.b64encode(contents).decode('utf-8')
                
                # Process
                result = explainer.explain(str(temp_path), symptoms="", threshold=0.42, age_group="Adult (40-64)")
                
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "prediction": result["prediction"],
                    "probability": float(result["probability"]),
                    "uncertainty": result["uncertainty"],
                    "explanation": result.get("explanation", ""),
                    "gradcam_image": result.get("gradcam_image"),
                    "original_image": original_image_base64,
                    "region": result.get("region", "Lung Field")
                })
                
                processed_count += 1
                
                # Send success update
                yield f"data: {json.dumps({'type': 'success', 'current': idx + 1, 'total': total_files, 'filename': file.filename, 'prediction': result['prediction'], 'probability': float(result['probability'])})}\n\n"
                await asyncio.sleep(0.1)
                
            except Exception as e:
                error_msg = str(e)[:200]  # Truncate long error messages
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)  # Full error in results file
                })
                yield f"data: {json.dumps({'type': 'error', 'current': idx + 1, 'filename': file.filename, 'error': error_msg})}\n\n"
                await asyncio.sleep(0.1)
        
        # Save results to a file that frontend can fetch
        persistent_dir = Path("batch_reports")
        persistent_dir.mkdir(exist_ok=True)
        results_file = persistent_dir / f"batch_results_{batch_id}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "batch_id": batch_id,
                "total_files": total_files,
                "processed": processed_count,
                "failed": total_files - processed_count,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }, f, indent=2)
        
        # Send completion with download URL for results
        yield f"data: {json.dumps({'type': 'complete', 'processed': processed_count, 'total': total_files, 'batch_id': batch_id, 'results_url': f'/batch_results/{batch_id}'})}\n\n"
        
        # Cleanup temp files
        shutil.rmtree(batch_temp, ignore_errors=True)
    
    return StreamingResponse(generate_progress(), media_type="text/event-stream")

@app.get("/batch_results/{batch_id}")
async def get_batch_results(batch_id: str):
    """Get batch processing results"""
    results_file = Path("batch_reports") / f"batch_results_{batch_id}.json"
    
    if not results_file.exists():
        return {"error": "Batch results not found"}
    
    with open(results_file, 'r') as f:
        return json.load(f)

@app.post("/batch_analyze")
async def batch_analyze(files: List[UploadFile] = File(...)):
    """Batch process multiple X-rays and return ZIP with individual PDF reports"""
    if explainer is None:
        return {"error": "Model failed to load"}
    
    if not files or len(files) == 0:
        return {"error": "No files uploaded"}
    
    if len(files) > 100:
        return {"error": "Maximum 100 files allowed per batch"}
    
    # Create temp directories
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_temp = Path("batch_temp") / batch_id
    batch_temp.mkdir(parents=True, exist_ok=True)
    
    reports_dir = batch_temp / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    results = []
    processed_count = 0
    
    try:
        for file in files:
            try:
                # Validate file
                allowed_ext = ['.jpg', '.jpeg', '.png', '.webp']
                file_ext = Path(file.filename).suffix.lower()
                if file_ext not in allowed_ext:
                    results.append({
                        "filename": file.filename,
                        "status": "error",
                        "error": f"Invalid file type: {file_ext}"
                    })
                    continue
                
                # Read and validate
                contents = await file.read()
                if len(contents) > 50 * 1024 * 1024:
                    results.append({
                        "filename": file.filename,
                        "status": "error",
                        "error": "File too large (max 50MB)"
                    })
                    continue
                
                # Validate image
                from PIL import Image
                import io as iolib
                try:
                    img = Image.open(iolib.BytesIO(contents))
                    img.verify()
                    img = Image.open(iolib.BytesIO(contents))
                    w, h = img.size
                    if w < 100 or h < 100 or w > 10000 or h > 10000:
                        results.append({
                            "filename": file.filename,
                            "status": "error",
                            "error": f"Invalid dimensions: {w}x{h}"
                        })
                        continue
                except Exception as e:
                    results.append({
                        "filename": file.filename,
                        "status": "error",
                        "error": f"Invalid image: {str(e)}"
                    })
                    continue
                
                # Save temp file
                temp_path = batch_temp / file.filename
                with open(temp_path, "wb") as buffer:
                    buffer.write(contents)
                
                # Process
                result = explainer.explain(str(temp_path), symptoms="", threshold=0.42, age_group="Adult (40-64)")
                
                # Generate PDF report
                pdf_filename = Path(file.filename).stem + "_report.pdf"
                pdf_path = reports_dir / pdf_filename
                
                generate_pdf_report(
                    pdf_path,
                    file.filename,
                    result["prediction"],
                    result["probability"],
                    result["uncertainty"],
                    result.get("explanation", ""),
                    result.get("gradcam_image")
                )
                
                results.append({
                    "filename": file.filename,
                    "status": "success",
                    "prediction": result["prediction"],
                    "probability": float(result["probability"]),
                    "report": pdf_filename
                })
                
                processed_count += 1
                
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "error": str(e)
                })
        
        # Create ZIP file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add all PDF reports
            for report_file in reports_dir.glob("*.pdf"):
                zip_file.write(report_file, report_file.name)
            
            # Add summary JSON
            summary = {
                "batch_id": batch_id,
                "total_files": len(files),
                "processed": processed_count,
                "failed": len(files) - processed_count,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }
            zip_file.writestr("batch_summary.json", json.dumps(summary, indent=2))
        
        # Cleanup temp files
        shutil.rmtree(batch_temp, ignore_errors=True)
        
        # Return ZIP
        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=tb_batch_reports_{batch_id}.zip"
            }
        )
        
    except Exception as e:
        # Cleanup on error
        shutil.rmtree(batch_temp, ignore_errors=True)
        return {"error": f"Batch processing failed: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)

