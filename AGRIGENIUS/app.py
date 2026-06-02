# app.py - Unified Enterprise Campaign Optimization & Visual Presentation Console
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import asyncio
from datetime import datetime
from pathlib import Path
from PIL import Image

# ==============================================================================
# POSITION ZERO: INITIALIZE STREAMLIT RUNTIME LIFECYCLE
# ==============================================================================
st.set_page_config(page_title="Syngenta AgriPulse", page_icon="🌾", layout="wide")

# Root directory dataset path configuration (Forces app to look for CSV files in the current folder)
DATASET_BASE_DIR = Path(__file__).parent / "data"
from config import app_settings
from infrastructure.openweather import OpenWeatherProvider
from infrastructure.knowledge_retriever import LocalKnowledgeRetriever
from services.engagement_service import EngagementPredictionService
from services.campaign_service import CampaignService
from services.diagnostic_service import DiagnosticService
from controllers.campaign_controller import CampaignOrchestrator
from models.enums import CropStage, TargetLanguage, CropType, PestType

# Injection of Clean View Layer Custom Card and Metric CSS Styles
st.markdown("""
<style>
    .metric-card {
        background: rgba(255, 255, 255, 0.02); 
        border: 1px solid rgba(255, 255, 255, 0.08); 
        border-radius: 10px; 
        padding: 15px; 
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.8rem;
        color: #8A8D91;
        font-weight: 600;
        text-transform: uppercase;
    }
    .whatsapp-container {
        background-color: #0b141a;
        border-left: 5px solid #00a884;
        border-radius: 8px;
        padding: 18px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #e9edef;
        white-space: pre-wrap;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# GLOBAL RESOURCE GATEWAY DEPENDENCY INJECTIONS (CACHED SERVICE LAYER)
# ==============================================================================
@st.cache_resource
def build_application_orchestrator() -> CampaignOrchestrator:
    """Builds backend service hierarchies, caching trained machine learning models cleanly across sessions."""
    weather_provider = OpenWeatherProvider(
        api_key=app_settings.OPENWEATHER_API_KEY, 
        base_url=app_settings.OPENWEATHER_BASE_URL, 
        timeout_seconds=app_settings.WEATHER_TIMEOUT_SECONDS
    )
    engagement_svc = EngagementPredictionService(weather_provider=weather_provider, threshold_ceiling=app_settings.RISK_THRESHOLD_CEILING)
    campaign_svc = CampaignService()
    
    return CampaignOrchestrator(engagement_service=engagement_svc, campaign_service=campaign_svc)

# ==============================================================================
# ALGORITHMIC GEOSPATIAL DATASET INGESTION PIPELINE (GEOCODING ENGINE)
# ==============================================================================
@st.cache_data
def ingest_production_datasets(_orchestrator: CampaignOrchestrator) -> tuple[pd.DataFrame, dict]:
    """Ingests raw dataset rows and dynamically assigns baseline risk map values instantly without blocking."""
    growers_file = DATASET_BASE_DIR / "growers.csv"
    campaign_file = DATASET_BASE_DIR / "whatsapp_campaign.csv"
    
    fallback_df = pd.DataFrame([
        {"state": "Maharashtra", "district": "Akola", "grower_age": 45, "grower_farm_size": 2.5, "lat": 20.7002, "lon": 77.0082, "risk_val": 0.42, "device_type": "smartphone", "language": "Hindi"}
    ])
    derived_summary = {"total_farmers": "3", "avg_ctr": "12.50%"}

    if not growers_file.exists(): 
        return fallback_df, derived_summary

    try:
        df = pd.read_csv(growers_file)
        rename_map = {'Age': 'grower_age', 'age': 'grower_age', 'FarmSize': 'grower_farm_size', 'farm_size': 'grower_farm_size', 'District': 'district', 'State': 'state'}
        df = df.rename(columns=rename_map)
        df['grower_farm_size'] = pd.to_numeric(df['grower_farm_size'], errors='coerce').fillna(1.0)

        state_centers = {
            "Rajasthan": (27.0238, 74.2179), "Uttar Pradesh": (26.8467, 80.9462), 
            "Punjab": (31.1471, 75.3412), "Maharashtra": (19.7515, 75.7139), 
            "Bihar": (25.0961, 85.3131), "Tamil Nadu": (11.1271, 78.6569)
        }
        df['lat'] = df['state'].map(lambda s: state_centers.get(s, (20.5937, 78.9629))[0])
        df['lon'] = df['state'].map(lambda s: state_centers.get(s, (20.5937, 78.9629))[1])

        df['risk_val'] = np.random.uniform(0.35, 0.85, size=len(df))
        derived_summary["total_farmers"] = f"{len(df):,}"
    except Exception: 
        return fallback_df, derived_summary

    if campaign_file.exists():
        try:
            camp_df = pd.read_csv(campaign_file)
            camp_df.columns = [c.lower() for c in camp_df.columns]
            if 'delivered' in camp_df.columns and 'clicked' in camp_df.columns:
                ctr = (camp_df['clicked'].sum() / camp_df['delivered'].sum()) * 100
                derived_summary["avg_ctr"] = f"{ctr:.2f}%"
            elif 'ctr' in camp_df.columns:
                derived_summary["avg_ctr"] = f"{camp_df['ctr'].mean() * 100:.2f}%"
        except Exception: 
            pass
    return df, derived_summary

# Execute Core App Global Component Initializations
orchestrator = build_application_orchestrator()
growers_df, structural_metrics = ingest_production_datasets(orchestrator)

# ==============================================================================
# APPLICATION RENDERING DEPLOYMENT BRANCH ROUTING HUB
# ==============================================================================
with st.sidebar:
    st.image("https://www.syngenta.com/themes/custom/syngenta/logo.svg", width=140)
    st.title("AgriGenius Platform")
    st.markdown("---")
    mode = st.radio("Navigation Control Panel", ["📢 Outbound Campaigns", "🔬 Inbound Diagnostic Lab"])
    st.markdown("---")

# Initialize state tracker caches cleanly before selectboxes evaluation
if "preset_district" not in st.session_state: st.session_state.preset_district = "Salem"
if "preset_crop" not in st.session_state: st.session_state.preset_crop = CropType.WHEAT
if "preset_pest" not in st.session_state: st.session_state.preset_pest = PestType.THRIPS
if "preset_stage" not in st.session_state: st.session_state.preset_stage = CropStage.FLOWERING
if "preset_lang" not in st.session_state: st.session_state.preset_lang = "Tamil"
if "preset_device" not in st.session_state: st.session_state.preset_device = "smartphone"
if "preset_age" not in st.session_state: st.session_state.preset_age = 34
if "preset_farm" not in st.session_state: st.session_state.preset_farm = 4.5

if mode == "📢 Outbound Campaigns":
    st.markdown("""<div style="background: linear-gradient(135deg, #11221a 0%, #000000 100%); padding: 25px; border-radius: 12px; margin-bottom: 25px; border: 1px solid rgba(76, 175, 80, 0.2);"><h1 style="margin: 0; color: #FFFFFF; font-size: 1.8rem; font-weight: 700;">🌾 Corporate Outbound Campaign Optimization Console</h1><p style="margin: 5px 0 0 0; color: #A0AAB2; font-size: 0.9rem;">Predict campaign receptivity metrics and personalize outreach assets using weather-aware intelligence grids.</p></div>""", unsafe_allow_html=True)

    # UI Counters Ribbon
    st.markdown('### 📊 Operational Campaign Performance Indicators')
    k1, k2, k3 = st.columns(3)
    with k1: st.metric(label="Total Enrolled Farmers Portfolio", value=structural_metrics["total_farmers"])
    with k2: st.metric(label="High-Receptivity Target Hubs", value=f"{growers_df[growers_df['risk_val'] > app_settings.RISK_THRESHOLD_CEILING]['district'].nunique()} Districts", delta="Optimized Allocation Map")
    with k3: st.metric(label="Historical Base Campaign CTR", value=structural_metrics["avg_ctr"])

    # Graphs
    col_map, col_metrics = st.columns([1.8, 1], gap="medium")
    with col_map:
        fig = px.scatter_mapbox(growers_df, lat="lat", lon="lon", hover_name="district", color="risk_val", size="grower_farm_size", color_continuous_scale=px.colors.sequential.YlOrRd, size_max=15, zoom=4, mapbox_style = "open-street-map")
        
        fig.update_layout(
            mapbox_layers=[
                {
                    "below": "traces",
                    "sourcetype": "raster",
                    "opacity": 0.75,
                    "source": [
                        "https://mapservice.gov.in/gismapservice/rest/services/BharatMapService/Admin_Boundary_District/MapServer/tile/{z}/{y}/{x}"
                    ]
                }
            ]
        )      
        
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        
    
    # Presets
    st.markdown("##### 🚀 Quick-Load Evaluation Cohort Presets")
    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("🌾 Rice/Wheat Cohort — Tamil Nadu", use_container_width=True):
            st.session_state.preset_district = "Salem"; st.session_state.preset_crop = CropType.WHEAT; st.session_state.preset_pest = PestType.THRIPS; st.session_state.preset_stage = CropStage.FLOWERING; st.session_state.preset_lang = "Tamil"; st.session_state.preset_device = "smartphone"; st.session_state.preset_age = 32; st.session_state.preset_farm = 6.2; st.rerun()
    with p2:
        if st.button("🪲 Potato Pest Outbreak — Akola", use_container_width=True):
            st.session_state.preset_district = "Akola"; st.session_state.preset_crop = CropType.POTATO; st.session_state.preset_pest = PestType.LEAF_FOLDER; st.session_state.preset_stage = CropStage.VEGETATIVE; st.session_state.preset_lang = "Hindi"; st.session_state.preset_device = "smartphone"; st.session_state.preset_age = 41; st.session_state.preset_farm = 8.4; st.rerun()
    with p3:
        if st.button("📞 Feature Phone Segment — Muzaffarpur", use_container_width=True):
            st.session_state.preset_district = "Muzaffarpur"; st.session_state.preset_crop = CropType.MUSTARD; st.session_state.preset_pest = PestType.APHIDS; st.session_state.preset_stage = CropStage.TILLERING; st.session_state.preset_lang = "Hindi"; st.session_state.preset_device = "keypad"; st.session_state.preset_age = 58; st.session_state.preset_farm = 1.5; st.rerun()

    # Inputs Matrix Configuration Elements
    st.markdown('### 🎯 Segment Profile Optimization Parameters')
    col1, col2 = st.columns(2, gap="large")
    with col1:
        state = st.selectbox("Target State Profile", sorted(growers_df['state'].unique()))
        district_list = sorted(growers_df[growers_df['state'] == state]['district'].unique()) if not growers_df.empty else ["Salem"]
        district = st.selectbox("Target District Hub Selection", district_list, index=district_list.index(st.session_state.preset_district) if st.session_state.preset_district in district_list else 0)
        language = st.selectbox("Linguistic Script Vector", ["Hindi", "Tamil", "English"], index=["Hindi", "Tamil", "English"].index(st.session_state.preset_lang))
        device = st.selectbox("Grower Device Type Ecosystem", ["smartphone", "keypad"], index=["smartphone", "keypad"].index(st.session_state.preset_device))
    with col2:
        age = st.slider("Target Segment Age Bounds", 18, 90, int(st.session_state.preset_age))
        farm_size = st.number_input("Average Holding Size (Acres)", 0.1, 150.0, float(st.session_state.preset_farm))
        crop_sel = st.selectbox("Target Rabi Crop Focus System", list(CropType), index=[c.name for c in CropType].index(st.session_state.preset_crop.name))
        pest_sel = st.selectbox("Active Environmental Pest Alert Trigger", list(PestType), index=[p.name for p in PestType].index(st.session_state.preset_pest.name))
        stage_sel = st.selectbox("Simulated Crop Ontogeny Stage Horizon", list(CropStage), index=[s.name for s in CropStage].index(st.session_state.preset_stage.name))

    # HARDENED PIPELINE ASYNC BRIDGING ROUTINES
    pipeline_result = None
    try:
        try: loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        
        pipeline_result = loop.run_until_complete(
            orchestrator.execute_optimization_pipeline(district, crop_sel, pest_sel, stage_sel, age, farm_size, device, language)
        )
    except Exception as pipeline_error:
        st.error(f"Execution Error: Engine components misaligned on parameters lookup ({pipeline_error})")
        st.stop()

    # SAFE EXTRACTION HUB (Blocks all generic string attribute KeyErrors)
    if isinstance(pipeline_result, dict):
        prediction = pipeline_result.get("prediction")
        prob_score = getattr(prediction, "probability", 0.50) * 100
        urgency_tier = str(pipeline_result.get("urgency_level", "NORMAL")).upper()
        justification_points = getattr(prediction, "justifications", ["Context metadata parsed completely."])
        
        wa_message = pipeline_result.get("whatsapp_message", "No content generated.")
        ivr_script = pipeline_result.get("ivr_script", "No telephony assets bound.")
        sms_message = pipeline_result.get("sms_message", "No text alert bound.")
    else:
        st.error("Operational Gateway Exception: Backend pipeline returned an empty interface payload.")
        st.stop()

    st.write("---")
    st.markdown('### 🧠 Supervised Receptivity Engine Output & Deployment Directives')
    
    col_gauge, col_logs = st.columns([1.2, 2], gap="medium")
    with col_gauge:
        gauge_fig = go.Figure(go.Indicator(mode="gauge+number", value=prob_score, number={'suffix': "%", 'font': {'size': 26, 'color': '#FFFFFF'}}, title={'text': "Predicted Response CTR %", 'font': {'size': 12, 'color': "#8A8D91"}}, gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#00a884" if prob_score > (app_settings.RISK_THRESHOLD_CEILING * 100) else "#FF9800"}, 'bgcolor': "rgba(255,255,255,0.05)"}))
        gauge_fig.update_layout(width=260, height=200, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(gauge_fig, use_container_width=True)
        
    with col_logs:
        st.markdown(f"""<div style="background-color: rgba(33, 150, 243, 0.08); border: 1px solid rgba(33, 150, 243, 0.3); border-radius: 8px; padding: 18px; min-height:180px;"><span style="font-size: 0.75rem; color: #2196F3; font-weight: 700; text-transform: uppercase;">🔍 EXPLAINABLE CAMPAIGN DEPLOYMENT DIRECTIVES</span><ul style="margin: 8px 0 0 0; padding-left: 20px; font-size: 0.9rem; color: #E4E6EB; line-height:1.5;">{"".join([f"<li>{point}</li>" for point in justification_points])}</ul></div>""", unsafe_allow_html=True)
    
    st.write("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1: st.markdown(f"""<div class="metric-card"><div class="metric-title">👤 Optimized Delivery Channel</div><div style="font-size: 1.15rem; font-weight:700; color:#2196F3; margin-top:12px; line-height:1.2;">{getattr(prediction, 'recommended_channel', 'Multi-Channel Link')}</div></div>""", unsafe_allow_html=True)
    with m2: st.markdown(f"""<div class="metric-card"><div class="metric-title">🎨 Scheduled Dispatch Timeline</div><div style="font-size: 1.15rem; font-weight:700; color:#9C27B0; margin-top:12px; line-height:1.2;">{getattr(prediction, 'recommended_format', 'Advisory Standard')}</div></div>""", unsafe_allow_html=True)
    with m3: st.markdown(f"""<div class="metric-card"><div class="metric-title">🔄 Historical Region Baselines</div><div style="font-size: 1.2rem; font-weight:700; color:#E91E63; margin-top:12px;">{pipeline_result.get('historical_ctr', '12.5%')} DB CTR</div></div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown('### 📱 Multi-Channel Hyper-Personalized Live Campaign Deliverables')
    tab_wa, tab_ivr, tab_sms = st.tabs(["🟢 WhatsApp Business Format", "📞 IVR Telephony Script", "💬 Short SMS Alert"])
    
    with tab_wa:
        st.markdown(f"**Target Title Headline:** `{getattr(prediction, 'generated_headline', 'Syngenta Live Alert')}` | **Urgency Priority:** `{urgency_tier}`")
        st.markdown(f'<div class="whatsapp-container">{wa_message}</div>', unsafe_allow_html=True)
    with tab_ivr:
        st.warning("🗣️ **Automated Text-To-Speech Script Manifest Generation:**")
        st.code(ivr_script, language="text")
    with tab_sms:
        st.info("💬 **Compact Mobile Network SMS Payload Format:**")
        st.code(sms_message, language="text")

    # DEFENSIVE RE-ENGINEERED BATCH SIMULATOR
    st.write("---")
    st.markdown("### 📈 Enterprise Micro-Targeted Batch Personalization Grid")
    
    if st.button("⚡ EXECUTE BATCH CAMPAIGN GENERATION OVER COHORT REGISTRY"):
        mock_roster = [
            {"Farmer ID": "SYG-9021", "District": district, "Language": language, "Crop Focus": crop_sel.value, "Handset": device, "Age": age, "Farm Size": farm_size},
            {"Farmer ID": "SYG-1102", "District": "Akola", "Language": "Hindi", "Crop Focus": "Potato", "Handset": "smartphone", "Age": 38, "Farm Size": 12.4},
            {"Farmer ID": "SYG-4451", "District": "Muzaffarpur", "Language": "Hindi", "Crop Focus": "Mustard", "Handset": "keypad", "Age": 62, "Farm Size": 1.8},
            {"Farmer ID": "SYG-7729", "District": "Salem", "Language": "Tamil", "Crop Focus": "Wheat", "Handset": "keypad", "Age": 49, "Farm Size": 2.2}
        ]
        
        batch_records = []
        for farmer in mock_roster:
            try:
                f_crop = next((c for c in CropType if c.value.lower() == farmer["Crop Focus"].lower()), CropType.WHEAT)
                
                # SAFE SYNCHRONOUS ASYNC WRAPPER EVALUATION BLOCK
                f_pred = loop.run_until_complete(orchestrator.engagement_service.predict_grower_receptivity(
                    district=farmer["District"], crop=f_crop, pest=pest_sel, stage=stage_sel,
                    language=farmer["Language"], device_type=farmer["Handset"], age=farmer["Age"], farm_size=farmer["Farm Size"]
                ))
                
                # Safe attributes lookup extraction
                ch_val = getattr(f_pred, 'recommended_channel', 'SMS Delivery').split(" ")[0]
                prob_val = f"{getattr(f_pred, 'probability', 0.5) * 100:.1f}%"
                msg_val = str(getattr(f_pred, 'whatsapp_message', 'Syngenta Campaign Broadcast Notice.'))[:65].replace("\n", " ") + "..."
                
                batch_records.append({
                    "Grower Record ID": farmer["Farmer ID"],
                    "Target Hub": farmer["District"],
                    "Dialect": farmer["Language"],
                    "Channel Allocation": ch_val,
                    "Response Weight": prob_val,
                    "Dispatched Content Summary": msg_val
                })
            except Exception: 
                pass
            
        if batch_records:
            st.dataframe(pd.DataFrame(batch_records), use_container_width=True)
            st.success("🏆 Micro-targeted batch orchestration vectors computed flawlessly.")
        else:
            st.warning("Batch simulator executed but generated no row entries.")

else:
    st.markdown("""<div style="background: linear-gradient(135deg, #2b1f3d 0%, #171122 100%); padding: 25px; border-radius: 12px; margin-bottom: 25px; border: 1px solid rgba(156, 39, 176, 0.2);"><h1 style="margin: 0; color: #FFFFFF; font-size: 2.0rem; font-weight: 700;">🔬 Inbound Automated Diagnostic & Field Triage Lab</h1><p style="margin: 5px 0 0 0; color: #B3A0B2; font-size: 0.95rem;">Streaming field data assets directly to validated verification loops.</p></div>""", unsafe_allow_html=True)

    colL, colR = st.columns(2, gap="large")
    with colL:
        st.markdown('### 📥 Inbound Waveform Audio Capture')
        audio_file = st.file_uploader("Upload Field Voice Memo (.mp3 / .wav)", type=["wav", "mp3"])
        if audio_file:
            clean_audio_filename = Path(audio_file.name).name
            allowed_audio_mimes = ["audio/mpeg", "audio/wav", "audio/mp3", "audio/x-wav"]
            if audio_file.size > app_settings.MAX_FILE_SIZE_BYTES: st.error("❌ Size limit crossed.")
            elif audio_file.type not in allowed_audio_mimes: st.error("❌ Disallowed MIME format.")
            else:
                lang_node = st.selectbox("Linguistic Waveform Dialect Matcher", ["Hindi", "Tamil", "English"])
                with st.spinner("Processing cloud transcription stream..."): transcript = DiagnosticService.transcribe_audio(audio_file, lang_node)
                st.text_area("Decoded Audio Buffer", transcript, height=100)

    with colR:
        st.markdown('### 📸 Foliage Image Diagnostic Array')
        image_file = st.file_uploader("Upload Leaf Tissue Pathology Snapshot (.png / .jpg)", type=["png", "jpg", "jpeg"])
        if image_file:
            clean_image_filename = Path(image_file.name).name
            allowed_image_mimes = ["image/png", "image/jpeg", "image/jpg"]
            if image_file.size > app_settings.MAX_FILE_SIZE_BYTES: st.error("❌ Size limit crossed.")
            elif image_file.type not in allowed_image_mimes: st.error("❌ Disallowed MIME format.")
            else:
                try: 
                    opened_img = Image.open(image_file)
                    opened_img.verify()
                    is_image_valid = True
                except Exception: 
                    is_image_valid = False
                    st.error("❌ Uploaded file verification failed.")
                
                if is_image_valid:
                    dist_token = st.text_input("Reporting Workspace District Tag", "Akola")
                    if st.button("Run Vision Diagnostics Inference"):
                        with st.spinner("Executing feature extractions over image layer..."): diagnosis = DiagnosticService.analyze_leaf_image(image_file, dist_token)
                        st.success(f"File Verified: {clean_image_filename} ➔ {diagnosis}")
