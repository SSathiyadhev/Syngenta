# config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class AppSettings(BaseSettings):
    """Encapsulates system settings and handles environment injection via Pydantic v2."""
    SIMULATION_MODE: bool = Field(default=True)
    CONFIDENCE_THRESHOLD: float = Field(default=0.70)
    RISK_THRESHOLD_CEILING: float = Field(default=0.60)
    MAX_FILE_SIZE_BYTES: int = Field(default=5_242_880)
    WEATHER_TIMEOUT_SECONDS: float = Field(default=4.0)
    
    OPENWEATHER_BASE_URL: str = Field(default="http://api.openweathermap.org/data/2.5/weather")
    
    OPENWEATHER_API_KEY: str = Field(default="", env="OPENWEATHER_API_KEY")
    GEMINI_API_KEY: str = Field(default="", env="GEMINI_API_KEY")
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )

# CRITICAL EXPLICIT INSTANTIATION:
settings = AppSettings()