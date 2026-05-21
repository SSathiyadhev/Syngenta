# infrastructure/openweather.py
import httpx
import asyncio
from interfaces.weather_provider import BaseWeatherProvider
from models import WeatherMetrics
from core.logger import setup_production_logger

logger = setup_production_logger("OpenWeatherProvider")

class OpenWeatherProvider(BaseWeatherProvider):
    """Live implementation of the weather interface executing asynchronous client network requests."""
    def __init__(self, api_key: str, base_url: str, timeout_seconds: float):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    async def fetch_live_metrics_async(self, district: str) -> WeatherMetrics:
        """Asynchronously queries the OpenWeather Map REST endpoint with exponential backoff retries."""
        if not self.api_key:
            logger.warning(f"No API key present for weather queries. Routing fail-safe baseline vectors for {district}.")
            return WeatherMetrics(temperature=27.5, humidity=55, rainfall=0.0, condition="clear")
            
        url = f"{self.base_url}?q={district},IN&appid={self.api_key}&units=metric"
        
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            retries = 3
            backoff_factor = 0.5
            
            for attempt in range(retries):
                try:
                    response = await client.get(url)
                    # Enforces strict HTTP status assertions over inbound payloads
                    response.raise_for_status()
                    data = response.json()
                    
                    return WeatherMetrics(
                        temperature=data["main"]["temp"],
                        humidity=data["main"]["humidity"],
                        rainfall=data.get("rain", {}).get("1h", 0.0),
                        condition=data["weather"][0]["main"].lower()
                    )
                except httpx.HTTPError as err:
                    logger.warning(f"Network transport fault on attempt {attempt + 1} for {district}: {str(err)}")
                    if attempt == retries - 1:
                        logger.error(f"Asynchronous telemetry queries completely exhausted for region {district}.")
                    await asyncio.sleep(backoff_factor * (2 ** attempt))
                    
        # Return elegant type-safe fallback if backend client is fully unreachable
        return WeatherMetrics(temperature=27.5, humidity=52, rainfall=0.0, condition="clear")