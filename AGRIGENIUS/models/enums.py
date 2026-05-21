# models/enums.py
from enum import Enum
from dataclasses import dataclass, field

class CropType(Enum):
    WHEAT = "Wheat"
    POTATO = "Potato"
    MUSTARD = "Mustard"
    CHICKPEA = "Chickpea"

class PestType(Enum):
    NONE = "None"
    THRIPS = "Thrips"
    STEM_BORER = "Stem Borer"
    APHIDS = "Aphids"
    LEAF_FOLDER = "Leaf Folder"

class CropStage(Enum):
    VEGETATIVE = "Vegetative"
    TILLERING = "Tillering"
    FLOWERING = "Flowering"
    HARVESTED = "Harvested"

class TargetLanguage(Enum):
    HINDI = "Hindi"
    TAMIL = "Tamil"
    ENGLISH = "English"

# 🟢 REINSTATE DELIVERY CHANNEL TO SATISFY PACKAGE CONTRACT
class DeliveryChannel(Enum):
    WHATSAPP = "WhatsApp Business API Platform"
    IVR = "Automated Interactive Voice XML Loop"
    SMS = "Short Message Service Protocol"

@dataclass
class EngagementPrediction:
    """The enterprise contract tracking true data-driven marketing receptivity metrics."""
    probability: float
    is_receptive: bool
    justifications: list[str] = field(default_factory=list)
    telemetry_stream: str = ""
    recommended_channel: str = "SMS Fallback"
    recommended_format: str = "Plain Text Standard"
    
    # Multi-Channel Generative Copy Assets
    generated_headline: str = ""
    whatsapp_message: str = ""
    ivr_script: str = ""
    sms_message: str = ""
    urgency_level: str = "normal"