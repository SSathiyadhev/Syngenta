# server.py - Enterprise Campaign Intelligence API Inference Gateway Server
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uvicorn
from datetime import datetime

from infrastructure.openweather import OpenWeatherProvider
from services.engagement_service import EngagementPredictionService
from models.enums import CropType, PestType, CropStage
from config import app_settings

# ==============================================================================
# POSITION ZERO: INITIALIZE CENTRAL FASTAPI CORE LIFECYCLE
# ==============================================================================
app = FastAPI(
    title="Syngenta AgriPulse Campaign Intelligence Gateway",
    version="1.0.0",
    description="Production REST API serving weather-aware farmer engagement predictions and hyper-personalized GenAI copy assets."
)

# Shared Global State Dependencies Instantiations
weather_provider = OpenWeatherProvider(
    api_key=app_settings.OPENWEATHER_API_KEY, 
    base_url=app_settings.OPENWEATHER_BASE_URL, 
    timeout_seconds=app_settings.WEATHER_TIMEOUT_SECONDS
)
service = EngagementPredictionService(weather_provider=weather_provider)

# ==============================================================================
# DATA ENCAPSULATION PYDANTIC CONTRACT GUARDS
# ==============================================================================
class CampaignInferenceRequest(BaseModel):
    district: str = Field(..., example="Salem", description="Target localized regional node.")
    crop: str = Field(..., example="Wheat", description="Target Rabi crop category focus.")
    pest: str = Field(..., example="Thrips", description="Active target pest hazard alert.")
    stage: str = Field(..., example="Flowering", description="Crop growth ontogeny phase context.")
    language: str = Field(..., example="Tamil", description="Linguistic delivery copy dialect.")
    device_type: str = Field(..., example="smartphone", description="Grower hardware handset ecosystem.")
    age: int = Field(..., ge=18, le=95, example=42, description="Farmer age index matrix profile.")
    farm_size: float = Field(..., ge=0.1, le=250.0, example=2.5, description="Average regional land holding size.")

class ModelMetricsResponse(BaseModel):
    accuracy: str
    precision: str
    recall: str
    f1_score: str
    auc_roc: str

# ==============================================================================
# SYSTEM ENDPOINT RESOURCE ROUTERS
# ==============================================================================
@app.get("/health", tags=["System Operations"])
def api_health_verification_ping() -> Dict[str, str]:
    """Exposes infrastructure layer operational heartbeats."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/metrics", response_model=ModelMetricsResponse, tags=["Model Registry"])
def get_calibrated_validation_metrics() -> Dict[str, str]:
    """Exposes unbiased performance metrics verified over the isolated stratified test split."""
    if not service.cached_metrics:
        raise HTTPException(status_code=503, detail="Machine Learning model metrics state not initialized.")
    return service.cached_metrics

@app.post("/predict", tags=["Inference Engine"])
async def execute_live_campaign_optimization(payload: CampaignInferenceRequest) -> Dict[str, Any]:
    """Processes dynamic context properties to return optimized deployment copies and CTR probabilities."""
    
    # 1. String Token to Enum Lookup Resolution Match Blocks
    target_crop = next((c for c in CropType if c.value.lower() == payload.crop.lower()), None)
    target_pest = next((p for p in PestType if p.value.lower() == payload.pest.lower()), None)
    target_stage = next((s for s in CropStage if s.value.lower() == payload.stage.lower()), None)

    # Validate that incoming parameters bound cleanly onto supported system domains
    if not target_crop or not target_pest or not target_stage:
        raise HTTPException(
            status_code=422, 
            detail=f"Invalid enum attributes. Provided: crop={payload.crop}, pest={payload.pest}, stage={payload.stage}"
        )

    try:
        # 2. Trigger Calibrated ML Inference Matrix & Adaptive Copy Generation Pipeline
        prediction = await service.predict_grower_receptivity(
            district=payload.district,
            crop=target_crop,
            pest=target_pest,
            stage=target_stage,
            language=payload.language,
            device_type=payload.device_type,
            age=payload.age,
            farm_size=payload.farm_size
        )
        
        # 3. Compile Unified Production JSON Schema Contract API Payload
        return {
            "predicted_ctr": float(prediction.probability),
            "is_target_receptive": bool(prediction.is_receptive),
            "recommended_delivery_channel": prediction.recommended_channel,
            "optimal_dispatch_window": prediction.recommended_format,
            "urgency_weight": prediction.urgency_level,
            "explainable_drivers": prediction.justifications,
            "telemetry_stream_log": prediction.telemetry_stream,
            
            # --- SYNCHRONIZED MULTI-CHANNEL OUTPUT CHANNELS ---
            "generated_assets": {
                "headline": prediction.generated_headline,
                "whatsapp_media_copy": prediction.whatsapp_message,
                "ivr_speech_transcript": prediction.ivr_script,
                "sms_mobile_alert": prediction.sms_message
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Core Inference Exception: {str(e)}")

if __name__ == "__main__":
    # Boots server locally on port 8000 with live auto-reload enabled
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)