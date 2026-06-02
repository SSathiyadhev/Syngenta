# test_pipeline.py - Production Systems Validation Benchmark
import asyncio
from infrastructure.openweather import OpenWeatherProvider
from services.engagement_service import EngagementPredictionService
from models.enums import CropType, PestType, CropStage
from config import app_settings

async def run_production_benchmark():
    print("========== 🧪 INITIALIZING ENTERPRISE ML ENGINE TEST ==========\n")
    
    # 1. Initialize Network Telemetry Providers
    weather_provider = OpenWeatherProvider(
        api_key=app_settings.OPENWEATHER_API_KEY, 
        base_url=app_settings.OPENWEATHER_BASE_URL, 
        timeout_seconds=app_settings.WEATHER_TIMEOUT_SECONDS
    )
    
    # 2. Spin up the Predictive Service Layer
    service = EngagementPredictionService(weather_provider=weather_provider)
    
    # 3. Print Serialized Training Assessment Verification
    print("📋 [TRAINING METRICS VERIFICATION]")
    print(f"➔ Evaluated Metrics Array: {service.cached_metrics}")
    print(f"➔ Processed Feature Weights: {service.feature_importances}\n")
    
    # 4. Scenario A: High-Affinity Demographic Profile Test
    print("📡 [SCENARIO A RUN: High-Affinity Smartphone Demographic]")
    result_a = await service.predict_grower_receptivity(
        district="Salem",
        crop=CropType.WHEAT,
        pest=PestType.THRIPS,
        stage=CropStage.FLOWERING,
        language="Tamil",
        device_type="smartphone",
        age=32,
        farm_size=8.5
    )
    print(f"➔ Output Score: {result_a.probability * 100:.2f}%")
    print(f"➔ Allocated Channel: {result_a.recommended_channel}")
    print(f"➔ Local Driver Trace: {result_a.justifications}\n")

    # 5. Scenario B: Low-Affinity Hardware Demographic Profile Test (Verify Variance)
    print("📡 [SCENARIO B RUN: Keypad / Low-Affinity Demographic]")
    result_b = await service.predict_grower_receptivity(
        district="Salem",
        crop=CropType.WHEAT,
        pest=PestType.THRIPS,
        stage=CropStage.VEGETATIVE,
        language="Tamil",
        device_type="keypad",
        age=68,
        farm_size=1.2
    )
    print(f"➔ Output Score: {result_b.probability * 100:.2f}%")
    print(f"➔ Allocated Channel: {result_b.recommended_channel}")
    print(f"➔ Local Driver Trace: {result_b.justifications}\n")
    
    print("================ 🧪 BENCHMARK COMPLETE ================")

if __name__ == "__main__":
    asyncio.run(run_production_benchmark())