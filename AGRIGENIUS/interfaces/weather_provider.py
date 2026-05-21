# interfaces/weather_provider.py
from abc import ABC, abstractmethod
from models import WeatherMetrics

class BaseWeatherProvider(ABC):
    """
    Abstract Base Class enforcing strict contract specifications 
    for real-time meteorological telemetry ingestion services.
    """
    
    @abstractmethod
    async def fetch_live_metrics_async(self, district: str) -> WeatherMetrics:
        """
        Asynchronously fetches and packages local environmental conditions.
        Must return a type-safe validated WeatherMetrics object instance.
        """
        pass