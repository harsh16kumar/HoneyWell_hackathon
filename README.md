# Honeywell Anomaly Detection

This project uses **Machine Learning (Random Forest Classifier)** and **Z-score based anomaly detection** to classify product quality and identify anomalies in sensor/feature data.

---

## 🚀 Features
- Trains a **Random Forest Classifier** to predict `Quality` from numeric features.
- Scales features with **StandardScaler**.
- Performs **Z-score anomaly detection**:
  - Computes per-feature deviations from the "correct" class (`Quality = 1`).
  - Extracts **top 7 most contributing features** for each anomaly.
  - Calculates a weighted **Anomaly Score (1–100%)** combining max and mean deviations.
- Saves:
  - `model.pkl` → trained Random Forest model
  - `scaler.pkl` → fitted StandardScaler
  - `stats.pkl` → per-feature mean & std for anomaly detection
  - `final_anomaly_output.csv` → merged dataset with predictions, anomaly scores, and top features
  - `final_anomaly_output.pkl` → same as above in pickle format

---

## 📂 Project Structure
