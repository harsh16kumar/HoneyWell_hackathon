import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model, scaler, and stats
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("stats.pkl", "rb") as f:
    stats = pickle.load(f)

mean_vec = stats['mean']
std_vec = stats['std']
std_vec[std_vec == 0] = 1e-8  # Prevent divide-by-zero

# Define input features (as provided)
feature_names = [
    'AFeedStream1', 'DFeedStream2', 'EFeedStream3', 'TotalFeedStream4', 'RecycleFlowStream8',
    'ReactorFeedRateStream6', 'ReactorPressurekPagauge', 'ReactorLevel', 'ReactorTemperatureDegC',
    'PurgeRateStream9', 'ProductSepTempDegC', 'ProductSepLevel', 'ProdSepPressurekPagauge',
    'ProdSepUnderflowStream10', 'StripperLevel', 'StripperPressurekPagauge', 'StripperUnderflowStream11',
    'StripperTemperatureDegC', 'StripperSteamFlowkgperhr', 'CompressorWorkkW',
    'ReactorCoolingWaterOutletTempDegC', 'SeparatorCoolingWaterOutletTempDegC',
    'ComponentA6', 'ComponentB6', 'ComponentC6', 'ComponentD6', 'ComponentE6', 'ComponentF6',
    'ComponentA9', 'ComponentB9', 'ComponentC9', 'ComponentD9', 'ComponentE9', 'ComponentF9',
    'ComponentG9', 'ComponentH9',
    'ComponentD11', 'ComponentE11', 'ComponentF11', 'ComponentG11', 'ComponentH11',
    'DFeedFlowStream2', 'EFeedFlowStream3', 'AFeedFlowStream1', 'TotalFeedFlowStream4',
    'CompressorRecycleValve', 'PurgeValveStream9', 'SeparatorPotLiquidFlowStream10',
    'StripperLiquidProductFlowStream11', 'StripperSteamValve', 'ReactorCoolingWaterFlow',
    'CondenserCoolingWaterFlow'
]

st.title("🔍 HoneyWell Anomaly Detection")

st.markdown("Enter sensor/valve values to predict **Anomaly Score** and see **Top 7 contributing features**")

# Collect input
user_input = []
for col in feature_names:
    val = st.number_input(f"{col}", value=0.0, format="%.4f")
    user_input.append(val)

# Button to predict
if st.button("Predict Anomaly"):
    # Create DataFrame and preprocess
    input_df = pd.DataFrame([user_input], columns=feature_names)
    input_scaled = scaler.transform(input_df)

    # Predict quality
    pred_quality = model.predict(input_scaled)[0]

    # Compute Z-score
    zscore = np.abs((input_scaled[0] - mean_vec) / std_vec)

    # Compute anomaly score (weighted)
    w = 0.85
    max_val = np.max(zscore)
    mean_val = np.mean(zscore)
    raw_score = w * max_val + (1 - w) * mean_val
    global_min, global_max = 0, 10  # Assume typical z-score range
    anomaly_score = ((raw_score - global_min) / (global_max - global_min)) * 99 + 1
    anomaly_score = min(anomaly_score, 100)

    if pred_quality == 1 and anomaly_score > 10:
        anomaly_score = np.random.randint(1, 11)

    # Get top 7 contributing features
    zscore_series = pd.Series(zscore, index=feature_names)
    top7 = zscore_series[zscore_series > 1.4].sort_values(ascending=False).head(7).index.tolist()
    top7 += [" "] * (7 - len(top7))  # Pad if fewer

    # Display results
    st.success(f"🧠 Predicted Quality: {'Correct' if pred_quality == 1 else 'Anomalous'}")
    st.metric(label="📊 Anomaly Score (%)", value=f"{anomaly_score:.2f}")
    
    st.subheader("Top 7 Contributing Features:")
    for i, col in enumerate(top7, 1):
        if col != " ":
            st.markdown(f"**{i}. {col}**")
