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
|
├── honey_well.csv
├── app.py
├── model.pkl
├── scaler.pkl
├── stats.pkl
├── final_anomaly_output.csv
├── final_anomaly_output.pkl
└── README.md


---

## ⚙️ Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/honeywell-anomaly-detection.git
   cd honeywell-anomaly-detection
2. Install Dependencies:
   ```bash
   pip install -r requirements.txt
3. Place your dataset:
   Ensure honey_well.csv is present in the root directory.
4. Run the script:
   ```bash
   python app.py

📊 Outputs:Classification Report is printed for model performance.
├──final_anomaly_output.csv includes:
├──Predicted_Quality
├──Anomaly_score% (1–100 scale)
├──top_feature_1 … top_feature_7 (most influential anomaly features)

Author - Harsh Kumar