# config.py - Centralized Application Control Switchboard
import os

# LIVE DEMO MODE SWITCH
# Set to False to activate the REAL live OpenAI Whisper API stream!
# If the venue Wi-Fi breaks during presentation, instantly flip back to True!
SIMULATION_MODE = False

# LIVE API TOKENS
# Replace 'your-actual-api-key-here' with your real OpenAI API key string
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-proj-6cJ4yDIdrH5RfLMBWf8vzMEgC29IK5teST6aPqGVVHGhLtWFenLq3xFub63Dxj2dWFas25AXqvT3BlbkFJZOOErBrnO7srUaoboDMPQTtM6dfIzhYqRfsKswiK9MMnbpeQVgCGsPrAd-bLzifEwBOF4Ai3UA")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCVHKHV4FGNstmo2lshkQfGZADoJ1uDIHg")
CONFIDENCE_THRESHOLD = 0.70
ISRO_NDVI_THRESHOLD = 0.35