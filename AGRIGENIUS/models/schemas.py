# models/schemas.py
from pydantic import BaseModel, Field, field_validator
from typing import List
from models.enums import DeliveryChannel

class WeatherMetrics(BaseModel):
    """Type-safe contract representing real-time weather client updates."""
    temperature: float = Field(..., description="Ambient air temperature in Celsius")
    humidity: int = Field(..., description="Relative humidity percentage")
    rainfall: float = Field(default=0.0, description="Precipitation depth metrics in mm")
    condition: str = Field(..., description="Dominant meteorological condition summary text")

class OutbreakPrediction(BaseModel):
    """Type-safe contract representing learned machine learning inference data output streams."""
    probability: float = Field(..., description="Calculated probability metric of vector outbreak")
    is_critical: bool = Field(..., description="Hazard threshold validation status flag")
    telemetry_stream: str = Field(..., description="String containing active weather client markers")
    justifications: List[str] = Field(..., description="XAI logs explaining machine learning feature weights")

class GrowerSegmentProfile(BaseModel):
    """Type-safe contract tracking demographic targeting selections and routing properties."""
    channel: DeliveryChannel = Field(..., description="Optimized communication channel destination router")
    format: str = Field(..., description="Predicted highest-converting media layout profile variant")
    timing: str = Field(..., description="Predicted peak user-activity response time delivery window")

class MarketingAssetPackage(BaseModel):
    """Type-safe contract forcing multi-channel validation checks on generative model outputs."""
    text: str = Field(..., description="WhatsApp alert copy script properties natively formatted")
    visual_concept: str = Field(..., description="Descriptive textual prompt designed to route to image canvases")
    video_script: str = Field(..., description="Scene storyboard directions for automated clip generation prompts")
    ivr_script: str = Field(..., description="Structured text properties built to stream to neural voice trunk networks")

    @field_validator("text")
    @classmethod
    def validate_non_empty_content(cls, value: str) -> str:
        """Type safety guardrail intercepting malformed or empty text asset properties early."""
        if not value.strip():
            raise ValueError("Generative AI engine returned an invalid empty text block asset.")
        return value