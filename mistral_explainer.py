# Mistral-based Explainer with RAG

import os
import base64
from pathlib import Path
import torch
import numpy as np
from mistralai import Mistral

from ensemble_models import load_ensemble
from preprocessing import LungPreprocessor, get_val_transforms
from qdrant_rag import QdrantRAG
import cv2

# .env is loaded by qdrant_rag module
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MistralExplainer:
    """Explainable AI system with Mistral LLM"""
    
    def __init__(self, model_path=None):
        self.model = load_ensemble(model_path, DEVICE)
        self.mistral = Mistral(api_key=MISTRAL_API_KEY) if MISTRAL_API_KEY else None
        self.rag = QdrantRAG()
        self.preprocessor = LungPreprocessor()
        
        if not self.mistral:
            print("⚠️  MISTRAL_API_KEY not set")
    
    def predict_with_uncertainty(self, image_path, n_samples=20):
        """Get prediction with uncertainty"""
        # Preprocess (stays grayscale)
        image = self.preprocessor.preprocess(image_path)
        
        # Transform — keep as grayscale, model expects 1 channel
        transforms = get_val_transforms()
        augmented = transforms(image=image)
        image_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
        
        # Ensure single channel
        if image_tensor.shape[1] == 3:
            image_tensor = image_tensor.mean(dim=1, keepdim=True)
        elif image_tensor.shape[1] != 1:
            image_tensor = image_tensor[:, :1, :, :]
        
        # MC Dropout prediction
        mean_prob, std_prob = self.model.predict_with_uncertainty(image_tensor, n_samples)
        
        mean_prob = mean_prob.item()
        std_prob = std_prob.item()
        
        # Uncertainty level
        # Relaxed uncertainty thresholds for hackathon presentation layer
        if std_prob < 0.15:
            uncertainty = "Low"
        elif std_prob < 0.25:
            uncertainty = "Medium"
        else:
            uncertainty = "High"
        
        return {
            "probability": mean_prob,
            "uncertainty_std": std_prob,
            "uncertainty_level": uncertainty,
            "image_tensor": image_tensor
        }
    
    def analyze_gradcam(self, image_tensor):
        """Analyze Grad-CAM heatmap"""
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        # Get Grad-CAM from DenseNet (primary model)
        target_layer = self.model.densenet.model.features.denseblock4
        cam = GradCAM(model=self.model.densenet, target_layers=[target_layer])
        
        grayscale_cam = cam(
            input_tensor=image_tensor,
            targets=[ClassifierOutputTarget(0)]
        )[0]
        
        # Analyze regions
        h = grayscale_cam.shape[0]
        upper = np.mean(grayscale_cam[:h//3])
        middle = np.mean(grayscale_cam[h//3:2*h//3])
        lower = np.mean(grayscale_cam[2*h//3:])
        
        regions = {"upper": upper, "middle": middle, "lower": lower}
        dominant = max(regions, key=regions.get)
        
        if dominant == "upper":
            region_desc = "upper lung zones (typical for post-primary TB)"
        elif dominant == "lower":
            region_desc = "lower lung zones"
        else:
            region_desc = "diffuse distribution across lung fields"
        
        return {
            "dominant_region": dominant,
            "description": region_desc,
            "heatmap": grayscale_cam
        }
    
    def create_gradcam_overlay(self, image_path, gradcam_heatmap):
        """Create a colored Grad-CAM overlay on the original X-ray, returned as base64 PNG"""
        original = cv2.imread(str(image_path))
        if original is None:
            return None
        
        h, w = original.shape[:2]
        
        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(gradcam_heatmap, (w, h))
        
        # Apply JET colormap for medical-grade visualization
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        
        # Blend: 60% original + 40% heatmap for clear overlay
        overlay = cv2.addWeighted(original, 0.6, heatmap_colored, 0.4, 0)
        
        # Encode to base64 PNG
        _, buffer = cv2.imencode('.png', overlay)
        return base64.b64encode(buffer).decode('utf-8')
    
    def transcribe_audio(self, audio_bytes):
        """Transcribe audio using Voxtral for voice-based symptom input"""
        if not self.mistral:
            return "Mistral API not configured"
        
        import tempfile
        import os
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
                
            with open(tmp_path, "rb") as f:
                response = self.mistral.audio.transcriptions.complete(
                    model="voxtral-mini-latest",
                    file={
                        "content": f,
                        "file_name": "audio.wav"
                    }
                )
            return response.text
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"⚠️ Voxtral transcription failed: {e}")
            return None
        finally:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def validate_symptoms(self, transcript):
        """Validates if transcribed symptoms relate to respiratory/TB using mistral-small-latest"""
        if not self.mistral or not transcript:
            return True # fail open if empty or no api
            
        prompt = f"""You are a medical triage routing filter.
Analyze the following transcribed patient symptoms:
"{transcript}"

Determine if these symptoms are EVEN REMOTELY related to respiratory issues, chest issues, lungs, tuberculosis, persistent fever, night sweats, coughing, or related systemic infections.
Return EXACTLY "VALID" if they are related or could be related.
Return EXACTLY "INVALID" if they clearly refer to completely unrelated issues (e.g., foot pain, dermatology, broken bones, eye infection)."""
        
        try:
            response = self.mistral.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )
            result = response.choices[0].message.content.strip().upper()
            return "VALID" in result
        except Exception as e:
            print(f"⚠️ Validation inference failed: {e}")
            return True # fail open
    
    def retrieve_evidence(self, prediction, region):
        """Retrieve medical evidence from RAG only when TB is suspected"""
        if prediction < 0.5:
            # Do not fetch evidence for normal/non-TB cases to avoid confusing clinical reasoning
            return []
            
        query = f"""
        Pulmonary tuberculosis chest x-ray findings,
        {region} consolidation cavitation,
        post-primary TB imaging patterns,
        WHO TB diagnostic imaging guidance
        """
        
        results = self.rag.query(query, top_k=4)
        return results
    
    def generate_explanation(self, prediction_data, gradcam_data, evidence, symptoms=None, age_group="Adult (40-64)", image_path=None):
        """Generate clinical explanation using Mistral function calling (tool use)"""
        if not self.mistral:
            return "Mistral API key not configured"
        
        prob = prediction_data["probability"]
        uncertainty = prediction_data["uncertainty_level"]
        uncertainty_std = prediction_data["uncertainty_std"]
        region = gradcam_data["description"]
        prediction_label = "Possible Tuberculosis" if prob >= 0.5 else "Likely Normal"
        symptoms_text = f"\nReported Symptoms: {symptoms}" if symptoms else "\nNo symptoms reported."
        
        # AGE-SPECIFIC PROBABILITY AND DEMOGRAPHIC INJECTION
        age_context = f"PATIENT DEMOGRAPHIC: {age_group}\n"
        if "Child" in age_group:
            age_context += "CRITICAL MEDICAL NOTE FOR CHILDREN (0-14): Pediatric TB is typically primary, pauci-bacillary, and non-cavitary. It frequently presents subtly as hilar lymphadenopathy without clear consolidation. Because AI models are adult-biased, ANY probability anomaly (e.g. >35-45%) in a child with symptoms is highly alarming and warrants aggressive triage. Do not look for cavities.\n"
        elif "Senior" in age_group:
            age_context += "CRITICAL MEDICAL NOTE FOR SENIORS (65+): Due to a blunted cell-mediated immune response, seniors typically present atypically. Cavitation is less common, while lower/mid-zone infiltrates mimicking common bacterial pneumonia are frequent. Therefore, lower confidence model outputs (e.g. 35-50%) cannot be dismissed if symptomatic, as the radiographic signature may just parallel basic pneumonia.\n"
        else:
            age_context += "MEDICAL NOTE: Standard Adult presentation (15-64 years) typically involves upper lobe consolidation or fibrocavitary lesions driven by an active immune response locking down the bacteria.\n"

        
        # Define tools for Mistral function calling
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "query_medical_evidence",
                    "description": "Search the WHO medical knowledge base for evidence related to a specific clinical query about tuberculosis. Returns relevant medical literature excerpts.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The medical query to search for, e.g. 'upper lobe cavitation tuberculosis diagnosis'"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "assess_uncertainty",
                    "description": "Get the detailed uncertainty analysis from the MC Dropout ensemble prediction, including confidence level and clinical reliability assessment.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "include_recommendation": {
                                "type": "boolean",
                                "description": "Whether to include a clinical recommendation based on uncertainty level"
                            }
                        },
                        "required": ["include_recommendation"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_clinical_guidelines",
                    "description": "Retrieve WHO clinical guidelines for TB screening recommendations based on specific findings or patient presentation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "finding_type": {
                                "type": "string",
                                "description": "Type of finding to look up guidelines for, e.g. 'abnormal_cxr', 'normal_cxr_with_symptoms', 'high_uncertainty'"
                            }
                        },
                        "required": ["finding_type"]
                    }
                }
            }
        ]
        
        user_text = f"""Analyze this TB screening result and provide a clinical explanation.
{age_context}
AI Model Output:
- Prediction: {prediction_label}
- TB Probability: {prob:.2%}
- Uncertainty (MC Dropout): {uncertainty} (std: {uncertainty_std:.4f})
- Grad-CAM Region: {region}
{symptoms_text}

Use your tools to:
1. Query relevant medical evidence for these specific findings
2. Assess the prediction uncertainty
3. Check applicable WHO clinical guidelines

Also, act as a 'Second Opinion' Vision Expert: Look at the attached Patient X-Ray image directly and provide your own visual assessment of the lungs, mentioning any opacities, lymphadenopathy, or clear areas, and compare it with the CNN's ({region}) findings.

Then provide a structured explanation with: Radiographic Alignment (including your direct Vision assessment), Clinical Correlation & Age Factors, Limitations, and Recomendations."""

        user_content = [{"type": "text", "text": user_text}]

        if image_path:
            import base64
            from PIL import Image
            import io
            
            try:
                # Open and compress the image
                with Image.open(image_path) as img:
                    # Convert to RGB if needed
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        
                    # Resize to safe dimensions for API payload (max 1024x1024)
                    img.thumbnail((768, 768), Image.Resampling.LANCZOS)
                    
                    # Save to memory buffer as compressed JPEG
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=75)
                    
                    # Encode to base64
                    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    
                    user_content.append({
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    })
            except Exception as e:
                print(f"⚠️ Failed to encode image for vision: {e}")

        # Initial message with clinical context
        messages = [
            {
                "role": "system",
                "content": """You are a TB clinical decision support agent. You have access to tools to query medical evidence and assess predictions.

WORKFLOW:
1. Analyze the AI prediction, patient data, and X-Ray image provided
2. Use your tools to gather relevant evidence and guidelines
3. Synthesize a structured clinical explanation

RULES:
- Use ONLY evidence retrieved from your tools
- Be concise and clinical (2-3 sentences per section)
- Focus on screening support, not definitive diagnosis
- Always recommend confirmatory testing for positive screens"""
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        # Step 1: Let Mistral decide which tools to call
        try:
            response = self.mistral.chat.complete(
                model="mistral-large-latest",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,
                max_tokens=2500
            )
        except Exception as e:
            print(f"⚠️ Mistral tool calling failed, falling back to direct generation: {e}")
            return self._generate_explanation_fallback(prob, uncertainty, region, evidence, symptoms)
        
        assistant_message = response.choices[0].message
        
        # Step 2: Process tool calls if any
        if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
            messages.append(assistant_message)
            
            for tool_call in assistant_message.tool_calls:
                func_name = tool_call.function.name
                import json
                try:
                    args = json.loads(tool_call.function.arguments)
                except:
                    args = {}
                
                # Execute the tool
                tool_result = self._execute_tool(func_name, args, prediction_data, gradcam_data, evidence)
                
                messages.append({
                    "role": "tool",
                    "name": func_name,
                    "content": tool_result,
                    "tool_call_id": tool_call.id
                })
            
            # Step 3: Get final synthesized response
            try:
                final_response = self.mistral.chat.complete(
                    model="mistral-large-latest",
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2500
                )
                return final_response.choices[0].message.content
            except Exception as e:
                print(f"⚠️ Final synthesis failed: {e}")
                return self._generate_explanation_fallback(prob, uncertainty, region, evidence, symptoms)
        
        # If no tool calls, return direct response
        return assistant_message.content or self._generate_explanation_fallback(prob, uncertainty, region, evidence, symptoms)
    
    def _execute_tool(self, func_name, args, prediction_data, gradcam_data, evidence):
        """Execute a tool call and return the result as a string"""
        
        if func_name == "query_medical_evidence":
            query = args.get("query", "tuberculosis chest x-ray findings")
            try:
                results = self.rag.query(query, top_k=3)
                if results:
                    return "\n\n".join([
                        f"[{r['source']}, Page {r['page']}] (Relevance: {r['score']:.2f}): {r['text'][:500]}"
                        for r in results
                    ])
                else:
                    return "No matching evidence found in knowledge base."
            except Exception as e:
                return f"Evidence retrieval failed: {e}"
        
        elif func_name == "assess_uncertainty":
            prob = prediction_data["probability"]
            std = prediction_data["uncertainty_std"]
            level = prediction_data["uncertainty_level"]
            region = gradcam_data["description"]
            
            assessment = f"Uncertainty Level: {level} (std={std:.4f})\n"
            assessment += f"TB Probability: {prob:.2%}\n"
            assessment += f"Model Attention: {region}\n"
            
            if args.get("include_recommendation", False):
                if level == "High":
                    assessment += "RECOMMENDATION: High uncertainty — prediction unreliable. Refer for specialist radiologist review."
                elif level == "Medium":
                    assessment += "RECOMMENDATION: Moderate uncertainty — consider additional clinical context and symptoms."
                else:
                    assessment += "RECOMMENDATION: Low uncertainty — model is confident in this prediction."
            
            return assessment
        
        elif func_name == "check_clinical_guidelines":
            finding_type = args.get("finding_type", "general")
            try:
                query_map = {
                    "abnormal_cxr": "WHO guidelines abnormal chest x-ray tuberculosis screening follow-up",
                    "normal_cxr_with_symptoms": "WHO guidelines normal chest x-ray TB symptoms further testing",
                    "high_uncertainty": "WHO recommendations uncertain TB screening results",
                    "general": "WHO tuberculosis screening chest x-ray guidelines recommendations"
                }
                query = query_map.get(finding_type, query_map["general"])
                results = self.rag.query(query, top_k=2)
                if results:
                    return "\n\n".join([
                        f"[WHO Guideline - {r['source']}, p.{r['page']}]: {r['text'][:500]}"
                        for r in results
                    ])
                else:
                    return "No specific guidelines found. Refer to latest WHO TB screening recommendations."
            except Exception as e:
                return f"Guidelines retrieval failed: {e}"
        
        return "Unknown tool called."
    

    def _generate_explanation_fallback(self, prob, uncertainty, region, evidence, symptoms):
        """Fallback: direct generation if function calling fails"""
        prediction_label = "Possible Tuberculosis" if prob >= 0.5 else "Likely Normal"
        evidence_text = "\n".join([f"[{r['source']}, p.{r['page']}]: {r['text'][:400]}" for r in evidence]) if evidence else "No evidence retrieved."
        symptoms_text = f"\nReported Symptoms: {symptoms}" if symptoms else ""
        
        prompt = f"""Provide a concise clinical explanation for this TB screening result.

AI Model Output: {prediction_label} (Probability: {prob:.2%}, Uncertainty: {uncertainty})
Grad-CAM: {region}
{symptoms_text}

Evidence: {evidence_text}

Structure: 1) Radiographic Alignment 2) Clinical Correlation & Age Factors 3) Limitations 4) Recommendation
Keep each section to 2-3 sentences."""

        response = self.mistral.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": "You are a TB screening clinical decision support assistant. Be concise and evidence-based."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )
        return response.choices[0].message.content
    
    def explain(self, image_path, symptoms=None, threshold=0.5, age_group="Adult (40-64)"):
        """Full explanation pipeline"""
        print(f"🔍 Analyzing: {image_path}\n")
        
        # Prediction with uncertainty
        pred_data = self.predict_with_uncertainty(image_path)
        
        # Grad-CAM analysis
        gradcam_data = self.analyze_gradcam(pred_data["image_tensor"])
        
        # Generate Grad-CAM overlay image
        gradcam_image = None
        try:
            gradcam_image = self.create_gradcam_overlay(image_path, gradcam_data["heatmap"])
        except Exception as e:
            print(f"⚠️ Grad-CAM overlay generation failed: {e}")
        
        # Retrieve evidence (graceful fallback)
        evidence = []
        try:
            evidence = self.retrieve_evidence(
                pred_data["probability"],
                gradcam_data["dominant_region"]
            )
        except Exception as e:
            print(f"⚠️ RAG evidence retrieval failed: {e}")
            evidence = [{"text": "Evidence retrieval unavailable", "source": "N/A", "page": 0, "score": 0}]
        
        # Generate explanation (graceful fallback)
        explanation = ""
        try:
            explanation = self.generate_explanation(
                pred_data,
                gradcam_data,
                evidence,
                symptoms,
                age_group=age_group,
                image_path=image_path
            )
        except Exception as e:
            print(f"⚠️ LLM explanation failed: {e}")
            prob = pred_data["probability"]
            region = gradcam_data["description"]
            explanation = (
                f"**Automated Analysis:** The model predicts {'Possible TB' if prob >= threshold else 'Normal'} "
                f"with {prob:.1%} confidence. Model attention focused on {region}. "
                f"Please consult a qualified healthcare professional for interpretation."
            )
        
        # Format output
        prediction_label = "Possible Tuberculosis" if pred_data["probability"] >= threshold else "Likely Normal"
        
        result = {
            "prediction": prediction_label,
            "probability": pred_data["probability"],
            "uncertainty": pred_data["uncertainty_level"],
            "uncertainty_std": pred_data["uncertainty_std"],
            "gradcam_region": gradcam_data["description"],
            "gradcam_image": gradcam_image,
            "evidence": evidence,
            "explanation": explanation
        }
        
        return result

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mistral_explainer.py <image_path> [symptoms]")
        print('Example: python mistral_explainer.py xray.png "cough, fever, weight loss"')
        sys.exit(1)
    
    image_path = sys.argv[1]
    symptoms = sys.argv[2] if len(sys.argv) > 2 else None
    
    explainer = MistralExplainer(model_path="models/ensemble_best.pth")
    result = explainer.explain(image_path, symptoms)
    
    # Print results
    print("="*60)
    print("TB SCREENING RESULT")
    print("="*60)
    print(f"\nPrediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.2%}")
    print(f"Uncertainty: {result['uncertainty']} (±{result['uncertainty_std']:.3f})")
    print(f"\nGrad-CAM Analysis: {result['gradcam_region']}")
    
    print("\n" + "="*60)
    print("CLINICAL EXPLANATION")
    print("="*60)
    print(f"\n{result['explanation']}")
    
    print("\n" + "="*60)
    print("EVIDENCE SOURCES")
    print("="*60)
    for i, ev in enumerate(result['evidence'], 1):
        print(f"\n{i}. {ev['source']} (Page {ev['page']}, Score: {ev['score']:.3f})")

if __name__ == "__main__":
    main()
