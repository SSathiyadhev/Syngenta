# controllers/campaign_controller.py
from services.content_generation_service import ContentGenerationService

class CampaignOrchestrator:
    """The central orchestration layer binding predictive models with multi-channel generative pipelines."""
    def __init__(self, engagement_service, campaign_service):
        self.engagement_service = engagement_service
        self.content_service = ContentGenerationService()
        self.campaign_service = campaign_service

    async def execute_optimization_pipeline(self, district, crop, pest, stage, age, farm_size, device, lang):
        # 1. Evaluate Supervised Machine Learning Weights
        prediction = await self.engagement_service.predict_grower_receptivity(
            district, crop, pest, stage, lang, device, age, farm_size
        )
        
        # 2. Extract Relative Canopy Telemetry Humidity
        try:
            live_humidity = float(prediction.telemetry_stream.split("Humidity: ")[1].split("%")[0])
        except Exception:
            live_humidity = 70.0

        # 3. Request Target Content Generation
        content = self.content_service.generate_campaign(
            crop=crop.value,
            pest=pest.value,
            stage=stage.value,
            district=district,
            language=lang,
            channel=prediction.recommended_channel,
            humidity=live_humidity,
            probability=prediction.probability
        )

        # 4. Defensive binding layer ensures properties exist on prediction tracker
        prediction.generated_headline = content.get("headline", "⚠️ AgriPulse Advisory")
        prediction.whatsapp_message = content.get("whatsapp", "No message variant generated.")
        prediction.ivr_script = content.get("ivr", "No interactive response script generated.")
        prediction.sms_message = content.get("sms", "No network SMS payload generated.")
        prediction.urgency_level = content.get("urgency", "LOW")

        # 5. Return explicit strings mapped identically to app.py presentation tabs
        return {
            "prediction": prediction,
            "historical_ctr": self.campaign_service.fetch_historical_db_ctr(district),
            "urgency_level": prediction.urgency_level,
            "whatsapp_message": prediction.whatsapp_message,
            "ivr_script": prediction.ivr_script,
            "sms_message": prediction.sms_message
        }