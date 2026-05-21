# services/content_generation_service.py
import os
import requests
from core.logger import setup_production_logger

logger = setup_production_logger("ContentGenerationService")

class ContentGenerationService:
    """Localization-aware campaign copy engine blending fallback templates with real-time Gemini LLM acceleration."""
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.fallback_matrix = {
            "tamil": {
                "alert": "⚠️ அவசர எச்சரிக்கை",
                "body": "உங்கள் {district} வட்டாரத்தில் உள்ள {crop} பயிர் தற்போது {stage} நிலையில் உள்ளதால், {pest} தாக்குதலுக்கு ஆளாகும் அபாயம் {risk_label} அளவில் உள்ளது.",
                "action": "உடனடி நடவடிக்கை: பாதுகாப்பு பூச்சிக்கொல்லி தெளிக்கவும்."
            },
            "hindi": {
                "alert": "⚠️ त्वरित चेतावनी",
                "body": "आपके {district} क्षेत्र में {crop} की फसल अभी {stage} चरण में है, जिससे {pest} संक्रमण का खतरा {risk_label} स्तर पर है।",
                "action": "त्वरित कार्रवाई: आगामी 48 घंटों के भीतर सुरक्षात्मक छिड़काव करें।"
            },
            "english": {
                "alert": "⚠️ Urgent Advisory",
                "body": "In your {district} hub, the {crop} system is currently at the {stage} phase, displaying a {risk_label} threat index for {pest} infestation.",
                "action": "Immediate Action: Apply targeted defensive crop protectants within 48 hours."
            }
        }

    def generate_campaign(self, **kwargs):
        """PRODUCTION GATEWAY: Immune to unexpected argument crashes and API dictionary anomalies."""
        crop = kwargs.get('crop', 'Crop')
        pest = kwargs.get('pest', 'Pest')
        stage = kwargs.get('stage', 'Stage')
        district = kwargs.get('district', 'Unknown')
        language = kwargs.get('language', 'English')
        channel = kwargs.get('channel', 'SMS')
        humidity = kwargs.get('humidity', 70.0)
        probability = kwargs.get('probability', 0.5)

        lang_clean = str(language).strip().lower()
        if lang_clean not in self.fallback_matrix: 
            lang_clean = "english"
        
        threat_index = (humidity / 100.0) * 0.4 + (probability * 0.6)
        urgency = "HIGH RISK" if threat_index > 0.68 else ("MEDIUM" if threat_index > 0.40 else "LOW")
        risk_label = "HIGH" if urgency == "HIGH RISK" else urgency

        # Live Generative Pipeline Attempt
        if self.api_key and len(str(self.api_key).strip()) > 5:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
                prompt = f"Write a crop protection advisory for a farmer in {district}. Crop: {crop}, Pest: {pest}, Stage: {stage}."
                
                response = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=5)
                res_json = response.json()
                
                # Defensively parse the nested JSON blocks to avoid KeyError hangups
                if response.status_code == 200 and "candidates" in res_json:
                    raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
                    return {
                        "headline": "⚠️ Gemini AI Alert",
                        "whatsapp": raw_text,
                        "ivr": raw_text,
                        "sms": raw_text[:140] if raw_text else "Alert",
                        "urgency": urgency
                    }
                else:
                    # Capture Google API error details safely without looking up raw ['message']
                    err_payload = res_json.get("error", {})
                    err_msg = err_payload.get("message", "Unknown API internal anomaly")
                    logger.warning(f"Gemini API handshake failed: {err_msg}. Routing to fallback matrix.")
            except Exception as e:
                logger.error(f"Generative network cluster timeout exception: {e}")

        # Deterministic Fallback Matrix execution pathway if API is unreachable
        vocab = self.fallback_matrix[lang_clean]
        body_context = vocab["body"].format(district=district, crop=crop, stage=stage, pest=pest, risk_label=risk_label)
        
        fallback_whatsapp = f"*{vocab['alert']}*\n\n{body_context}\n\n🎯 *{vocab['action']}*"
        fallback_ivr = f"Attention farmer, advisory for {district} regarding {crop} and {pest}. Risk level evaluates as {risk_label}."
        fallback_sms = f"Syngenta Alert: {crop} at {risk_label} risk in {district} hub."

        return {
            "headline": f"⚠️ Static Advisory [{district}]",
            "whatsapp": fallback_whatsapp,
            "ivr": fallback_ivr,
            "sms": fallback_sms,
            "urgency": urgency
        }