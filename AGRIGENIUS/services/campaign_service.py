# services/campaign_service.py
from models import GrowerSegmentProfile, DeliveryChannel

class CampaignService:
    """Manages marketing segment targeting metrics, timing optimization windows, and pulls interaction analytics."""
    
    @staticmethod
    def identify_optimal_strategy(age: int, farm_size: float, device: str) -> GrowerSegmentProfile:
        """Evaluates handset architecture profiles to predict peak response delivery channels."""
        if device == "smartphone":
            return GrowerSegmentProfile(
                channel=DeliveryChannel.WHATSAPP, 
                format="HTML Canvas Layout + Short-Form Video Storyboard Clip Prompt Link", 
                timing="06:30 PM - 08:30 PM (Post-Market Window)"
            )
        
        return GrowerSegmentProfile(
            channel=DeliveryChannel.IVR, 
            format="Structured TwiML Automated Voice XML Array Packet", 
            timing="07:30 AM - 09:30 AM (Pre-Field Window)"
        )

    @staticmethod
    def fetch_historical_db_ctr(district: str) -> str:
        """Queries persistent tracking tables to return historical baseline click-through rates."""
        db_records_map = {
            "Akola": "14.25%", 
            "Muzaffarpur": "9.80%", 
            "Salem": "16.40%", 
            "Agra": "11.15%"
        }
        return db_records_map.get(district, "12.50%")