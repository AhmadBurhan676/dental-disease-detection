# 🦷 DentalVision AI — Panoramic Dental Radiography Pathology Detection

Automated Deep Learning Multi-Disease Detection for Panoramic Dental Radiographs (OPGs) using the benchmark dataset curated by **Orvile** on Kaggle (*Román et al., Sensors 2021*).

---

## 🎯 Target Pathologies Detected
1. **Dental Caries (Demineralization):** Coronal & interproximal enamel demineralization cavities.
2. **Periapical Lesions (Apical Periodontitis):** Radiolucent periapical lesions and lamina dura disruptions.
3. **Periodontitis (Alveolar Bone Loss):** Horizontal & vertical alveolar crest resorption (>2mm from CEJ).

---

## 🚀 Key Features
- **Interactive Web App (Streamlit & FastAPI):** Real-time diagnostic prediction studio.
- **Explainable AI (Grad-CAM):** Multi-layer attention maps (Composite, Caries, Periapical, Periodontitis).
- **Radiograph Enhancement Tools:** Built-in CLAHE contrast optimization and negative view toggle.
- **Model Metrics & Confusion Matrix:** Benchmarked on 598 clinical panoramic radiographs (Owandy I-max Touch).
- **Dataset Encyclopedia:** Full background on the Kaggle / Zenodo 4457648 dataset.

---

## 💻 Streamlit Deployment
- **Main file:** `streamlit_app.py`
- **Dependencies:** `requirements.txt`
