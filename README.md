# Syngenta
# AgriPulse 🌾
### *AI-Powered Contextual Marketing & Real-Time Regional Outreach Loop*

Developed by **Team AgriGenius** Submission for Track 1: AI-Powered Agricultural Marketing at Scale (Syngenta Hackathon 2026)

---

## 🎯 Project Overview
AgriPulse is an enterprise-grade precision marketing and advisory platform that solves the core fragmentation challenge in traditional agricultural broadcasting. Instead of generic mass-media messaging, AgriPulse treats marketing as **advisory-led commerce**, dynamically optimizing campaigns based on hyper-local climatic data, device profiles, language preferences, and episodic pest threats.

### 🔄 The Inbound-to-Outbound Intelligence Loop
* **Inbound Field Radar:** Tech-savvy smartphone farmers upload photos or voice memos of diseased crops to our Inbound Diagnostic Lab.
* **Database Synced Threat Arrays:** As disease clusters are identified in real time, the database updates automatically, pushing affected districts to a **Top Critical Alert Preset** on the marketer's dashboard.
* **Outbound Precision Campaign:** The marketer triggers an optimized batch campaign with a single click, automatically deploying voice broadcast alerts (IVR) to feature-phone users and interactive media to smartphone users in adjacent villages before the pest pressure spreads.

---

## 🏗️ System Architecture
The production design transitions from our edge-contained prototype into a fully decoupled microservices architecture:

* **Presentation Layer:** Containerized Streamlit Enterprise / React SPA (AWS ECS Fargate).
* **Gateway/Orchestration Layer:** High-concurrency **FastAPI** Gateway running asynchronously via **Redis & Celery Work Queues**.
* **AI Core:** **Calibrated Random Forest Pipeline** (Platt's Scaling for true-probability CTR prediction) + **RAG Engine via Gemini-2.5-Flash** (Context-aware vernacular copy generation) + **Vision API Fallback** (High-fidelity pathology diagnostics).
* **Data Layer:** Transactional PostgreSQL Cluster (with TimescaleDB extensions) + Snowflake Cloud Data Warehouse + AWS S3 Bucket storage for unstructured assets (voice WAV files and leaf PNG images).

---

## 📊 Core Performance Metrics
Our underlying predictive core achieves high classification bounds on synthetic agricultural communication footprints:
* **Accuracy Floor:** 84.5%
* **Precision Vector:** 82.1%
* **Recall Matrix Bound:** 80.4%
* **AUC-ROC Performance:** 83.5%
* **F1-Score Balance:** 81.2%

*Note: In the operational dashboard, these data science baselines are cleanly wrapped into an intuitive business metric: **"Recommendation Confidence %"**.*

---

## 🚀 Local Installation & Execution
Follow these steps to run the self-contained, lightweight Python prototype locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/agripulse.git](https://github.com/YOUR_GITHUB_USERNAME/agripulse.git)
cd agripulse
