# services/content_service.py
import json
from models import MarketingAssetPackage, CropType, PestType, CropStage
from interfaces.vector_retriever import BaseVectorRetriever
from core.logger import setup_production_logger

logger = setup_production_logger("ContentGenService")

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

class ContentService:
    """Orchestrates dynamic text transformation pipelines through validated LLM system prompts."""
    def __init__(self, api_key: str, simulation_mode: bool, vector_store: BaseVectorRetriever):
        self.simulation_mode = simulation_mode or not HAS_GEMINI
        self.vector_store = vector_store
        if not self.simulation_mode and HAS_GEMINI:
            try:
                self.client = genai.Client(api_key=api_key)
                self.model_name = 'gemini-1.5-flash'
            except Exception as e:
                logger.exception(f"Failed to initialize live Gemini Client. Forcing simulation mode: {str(e)}")
                self.simulation_mode = True

    def assemble_targeted_media_kit(self, lang: str, district: str, crop: CropType, pest: PestType, product: str, stage: CropStage, telemetry_summary: str) -> MarketingAssetPackage:
        """Queries local vector store manuals and compiles structural context blocks to generate localized assets."""
        rag_context = self.vector_store.similarity_search(crop, pest)
        
        if self.simulation_mode:
            return MarketingAssetPackage(**self._fetch_fallback_templates(lang, district, crop, pest, product))
            
        system_prompt = f"""
        You are an expert agricultural copywriter for Syngenta India.
        Generate an optimized campaign asset package for growers in {district} cultivation lines tracking {crop.value} ({stage.value} phase).
        Threat Trigger: {pest.value}. Atmospheric Data: {telemetry_summary}. Product solution: {product}.
        Legal Safety Guardrail Constraint from Vector Memory: {rag_context}
        LIGNUISTIC REQUIREMENT: Write all creative copy properties natively using the typographic script of: {lang}.
        
        Return your response matching this exact, raw valid JSON mapping model configuration:
        {{
            "text": "WhatsApp broadcast copy string in localized script containing contextual emojis.",
            "visual_concept": "Detailed textual canvas asset prompt instructions describing the layout composition matrix for an ad banner graphics container.",
            "video_script": "30-second localized scene description tracking storyboard prompt structures.",
            "ivr_script": "Spoken word narrative text properties mapped to structural neural text-to-speech network broadcast arrays."
        }}
        Do not wrap your output inside markdown formatting blocks or text backticks. Return raw pure JSON only.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=system_prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json")
            )
            return MarketingAssetPackage(**json.loads(response.text.strip()))
        except Exception as e:
            logger.exception(f"Generative API failure. Deploying fallback templates: {str(e)}")
            return MarketingAssetPackage(**self._fetch_fallback_templates(lang, district, crop, pest, product))

    def _fetch_fallback_templates(self, lang: str, district: str, crop: CropType, pest: PestType, product: str) -> dict:
        if lang.lower() == "tamil":
            return {
                "text": f"🌾 அவசர எச்சரிக்கை! {district} பகுதியில் {crop.value} பயிரில் {pest.value} கண்டறியப்பட்டுள்ளது. சினஜென்டா {product} ஐப் பயன்படுத்தவும்!",
                "visual_concept": f"விளம்பரப் படம்: {district} கிராமப்புறத்தில் ஒரு விவசாயி தனது {crop.value} வயலைச் சோதிப்பது, {pest.value} புழுக்கள்.",
                "video_script": f"வீடியோ காட்சி: சின்ஜென்டா {product} மூலம் உங்கள் பயிரை பாதுகாக்கவும்.",
                "ivr_script": f"வணக்கம் {district} விவசாயியே. உங்கள் பயிரில் {pest.value} தாக்கம் கண்டறியப்பட்டுள்ளது. 1ஐ அழுத்தவும்."
            }
        else:
            return {
                "text": f"🌾 किसान साथी, {district} जिले में आपकी {crop.value} फसल पर {pest.value} का प्रकोप सक्रिय है। तुरंत {product} का उपयोग करें!",
                "visual_concept": f"विज़ुअल अवधारणा: {district} के ग्रामीण परिदृश्य में एक किसान अपने {crop.value} खेत का निरीक्षण कर रहा है, पत्तियों पर {pest.value}।",
                "video_script": f"वीडियो स्क्रिप्ट: सिंजेंटा {product} से अपनी फसल बचाएं।",
                "ivr_script": f"नमस्ते किसान भाई। {district} क्षेत्र में आपकी {crop} फसल पर {pest.value} का हमला बढ़ रहा है। अधिक जानकारी के लिए 1 दबाएं।"
            }