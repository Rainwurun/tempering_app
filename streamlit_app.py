import streamlit as st
import numpy as np
import pandas as pd
import joblib
import io

# Load model and scalers
model = joblib.load("best_stacking_svr_bayes_model.pkl")
x_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Language selection
language = st.sidebar.radio("Language / 语言", ["English", "中文"])

# Labels (EN & CN)
labels = {
    "title": {"English": "Tempered Hardness Prediction for Steels", "中文": "钢材回火硬度预测"},
    "desc": {
        "English": "Input heat treatment parameters and alloy composition. Predict the post-tempering hardness (HRC).",
        "中文": "请输入热处理参数与合金成分，预测回火后的硬度（HRC）。"
    },
    "predict_btn": {"English": "Predict Tempered Hardness", "中文": "预测回火后硬度"},
    "reset_btn": {"English": "Reset Inputs", "中文": "重置输入"},
    "history_title": {"English": "Prediction History", "中文": "预测历史"},
    "export_btn": {"English": "Download Results as CSV", "中文": "下载预测结果 CSV"},
    "upload_title": {"English": "Batch Prediction from CSV", "中文": "CSV 批量预测"},
    "upload_hint": {"English": "Upload a CSV with 7 columns: Temp, C, Cr, Time, Si, Mo, Mn", "中文": "上传含7列（Temp, C, Cr, Time, Si, Mo, Mn）的CSV文件"},
}

# Show title and description
st.title(labels["title"][language])
st.write(labels["desc"][language])

# Defaults
defaults = {
    "temperature": 500, "time": 600, "C": 0.50, "Cr": 0.50, "Si": 0.50, "Mo": 0.50, "Mn": 0.50
}
tooltips = {
    "temperature": "Recommended range: 300–750 ºC",
    "time": "Typical industrial range: 100–10000 s",
    "C": "Common range: 0.2–1.0 wt%",
    "Cr": "Used for hardenability and corrosion resistance",
    "Si": "Strengthening through solid solution",
    "Mo": "Improves tempering resistance",
    "Mn": "Enhances hardenability"
}

# Function: slider + number input
def slider_with_input(label, min_val, max_val, default_val, step, unit="", tooltip=""):
    st.markdown(f"**{label}** ({unit})")
    if tooltip:
        st.caption(f"💡 {tooltip}")
    cols = st.columns([3, 2])
    slider_val = cols[0].slider("", min_val, max_val, value=default_val, step=step, key=f"{label}_slider")
    input_val = cols[1].number_input("", min_val, max_val, value=slider_val, step=step, key=f"{label}_input")
    return input_val

# Reset button
if st.button(labels["reset_btn"][language]):
    st.rerun()

# Input sections
st.subheader("📌 " + ("Tempering Parameters" if language == "English" else "回火参数"))
temperature = slider_with_input("Tempering Temperature", 0, 950, defaults["temperature"], 1, "ºC", tooltips["temperature"])
time = slider_with_input("Tempering Time", 0, 100000, defaults["time"], 10, "s", tooltips["time"])

st.subheader("🧪 " + ("Alloy Composition" if language == "English" else "合金成分"))
C = slider_with_input("C", 0.00, 5.00, defaults["C"], 0.01, "%wt", tooltips["C"])
Cr = slider_with_input("Cr", 0.00, 5.00, defaults["Cr"], 0.01, "%wt", tooltips["Cr"])
Si = slider_with_input("Si", 0.00, 5.00, defaults["Si"], 0.01, "%wt", tooltips["Si"])
Mo = slider_with_input("Mo", 0.00, 5.00, defaults["Mo"], 0.01, "%wt", tooltips["Mo"])
Mn = slider_with_input("Mn", 0.00, 5.00, defaults["Mn"], 0.01, "%wt", tooltips["Mn"])

# Predict single
if st.button(labels["predict_btn"][language]):
    X_input = np.array([[temperature, C, Cr, time, Si, Mo, Mn]])
    X_scaled = x_scaler.transform(X_input)
    y_scaled_pred = model.predict(X_scaled)
    y_pred = y_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1))
    result = round(y_pred[0][0], 2)
    st.success(f"{labels['predict_btn'][language]}: {result} HRC")
    st.session_state.history.append((temperature, time, C, Cr, Si, Mo, Mn, result))

# Show history
if st.session_state.history:
    st.subheader("📜 " + labels["history_title"][language])
    df_hist = pd.DataFrame(st.session_state.history, columns=[
        "Temp (ºC)", "Time (s)", "C", "Cr", "Si", "Mo", "Mn", "Predicted HRC"
    ])
    st.dataframe(df_hist, use_container_width=True)

    # Export button
    csv = df_hist.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=labels["export_btn"][language],
        data=csv,
        file_name="tempered_hardness_predictions.csv",
        mime="text/csv"
    )

# Batch upload
st.subheader("📂 " + labels["upload_title"][language])
st.caption(labels["upload_hint"][language])
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    try:
        df_input = pd.read_csv(uploaded_file)
        if df_input.shape[1] != 7:
            st.error("CSV must have exactly 7 columns.")
        else:
            X_raw = df_input.values
            X_scaled = x_scaler.transform(X_raw)
            y_scaled = model.predict(X_scaled)
            y_pred = y_scaler.inverse_transform(y_scaled.reshape(-1, 1))
            df_input["Predicted HRC"] = y_pred
            st.success("Batch prediction completed!")
            st.dataframe(df_input, use_container_width=True)

            # Download result
            out_csv = df_input.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Batch Results as CSV",
                data=out_csv,
                file_name="batch_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error: {e}")
