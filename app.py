"""
Mobile Price Prediction System - Streamlit web app.

Uses the project's final model: Gradient Boosting with the tuned settings
(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42).
The model is trained once from mobile_price_cleaned.csv when the app starts.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- settings
DATA_FILE = Path(__file__).parent / "mobile_price_cleaned.csv"
CURRENCY = "₹"   # prices in the dataset are in Indian rupees
TEAM = "Rahul Gupta, Pragati Bangwal, Aarya Awasthi, Alok Singh"

ALL_FEATURES = ["Ratings", "RAM", "ROM", "Mobile_Size",
                "Primary_Cam", "Selfi_Cam", "Battery_Power"]
SPEC_FEATURES = [f for f in ALL_FEATURES if f != "Ratings"]
GB_PARAMS = dict(n_estimators=200, learning_rate=0.1, max_depth=3,
                 random_state=42)

# Test-set results from the project (Chapter 7 of the report)
MODEL_INFO = {
    "full": {"label": "Full model (7 features, with rating)", "r2": 0.788},
    "spec": {"label": "Specification-only model (6 features)", "r2": 0.687},
}
# Typical absolute error of the final model by price band (test set)
SEGMENTS = [(970, "Budget", 729), (1499, "Mid-range", 1004),
            (15999, "Premium", 3321), (np.inf, "Flagship", 8678)]

st.set_page_config(page_title="Mobile Price Predictor", page_icon="📱",
                   layout="centered")


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner="Training the model...")
def train_models(path: str):
    df = pd.read_csv(path).drop_duplicates().reset_index(drop=True)
    y = df["Price"]
    full = make_pipeline(StandardScaler(), GradientBoostingRegressor(**GB_PARAMS))
    spec = make_pipeline(StandardScaler(), GradientBoostingRegressor(**GB_PARAMS))
    full.fit(df[ALL_FEATURES], y)
    spec.fit(df[SPEC_FEATURES], y)
    return full, spec, df


def options(df, col):
    """Sorted, positive values seen in the data for a select box."""
    vals = sorted(v for v in df[col].dropna().unique() if v > 0)
    return [int(v) if float(v).is_integer() else float(v) for v in vals]


def pick_default(opts, preferred):
    return opts.index(preferred) if preferred in opts else len(opts) // 2


def money(x):
    return f"{CURRENCY}{x:,.0f}"


# ---------------------------------------------------------------- page
st.title("📱 Mobile Price Prediction System")
st.caption("Estimate a smartphone's price from its specifications "
           "using a tuned Gradient Boosting model.")

if not DATA_FILE.exists():
    st.error("`mobile_price_cleaned.csv` was not found. Put the dataset file "
             "in the same folder as `app.py` and redeploy.")
    st.stop()

full_model, spec_model, df = train_models(str(DATA_FILE))

on_market = st.toggle(
    "This phone is already on the market (it has a user rating)", value=True,
    help="New phones have no rating yet, so the app switches to the "
         "specification-only model.")

with st.form("specs"):
    col1, col2 = st.columns(2)

    ram_opts, rom_opts = options(df, "RAM"), options(df, "ROM")
    cam_opts, selfie_opts = options(df, "Primary_Cam"), options(df, "Selfi_Cam")

    with col1:
        ram = st.selectbox("RAM (GB)", ram_opts, index=pick_default(ram_opts, 6))
        rom = st.selectbox("Storage / ROM (GB)", rom_opts,
                           index=pick_default(rom_opts, 64))
        size = st.number_input("Display size (inches)", min_value=2.0,
                               max_value=8.0, value=6.0, step=0.1)
        battery = st.slider("Battery (mAh)", min_value=1000, max_value=6000,
                            value=4000, step=100)
    with col2:
        primary = st.selectbox("Rear camera (MP)", cam_opts,
                               index=pick_default(cam_opts, 48))
        selfie = st.selectbox("Front camera (MP)", selfie_opts,
                              index=pick_default(selfie_opts, 8))
        rating = st.slider("User rating", min_value=2.8, max_value=4.8,
                           value=4.2, step=0.1, disabled=not on_market)

    submitted = st.form_submit_button("Predict price", type="primary",
                                      use_container_width=True)

if submitted:
    row = {"Ratings": rating, "RAM": ram, "ROM": rom, "Mobile_Size": size,
           "Primary_Cam": primary, "Selfi_Cam": selfie,
           "Battery_Power": battery}
    if on_market:
        key, model, cols = "full", full_model, ALL_FEATURES
    else:
        key, model, cols = "spec", spec_model, SPEC_FEATURES

    price = max(float(model.predict(pd.DataFrame([row])[cols])[0]), 0.0)
    band, err = next((name, e) for limit, name, e in SEGMENTS if price <= limit)

    st.metric("Estimated price", money(price))
    st.write(f"Price band: **{band}**. In testing, the model's typical error "
             f"for this band was about **±{money(err)}**.")
    st.caption(f"Model used: {MODEL_INFO[key]['label']}, "
               f"test R² = {MODEL_INFO[key]['r2']}.")

with st.expander("About this project"):
    st.markdown(
        f"""
- **Data:** {len(df)} smartphone listings with 7 numeric features
  (rating, RAM, ROM, display size, rear camera, front camera, battery).
- **Model:** Gradient Boosting (200 trees, learning rate 0.1, max depth 3),
  chosen from 10 model variants. Test MAE 3,459.10, RMSE 10,034.19, R² 0.788.
- **Two modes:** the full model uses the user rating; the specification-only
  model is for new phones that have no rating yet.
- **Limitations:** small dataset, no brand / processor / OS data, and large
  errors for flagship phones. Treat the result as an estimate, not a fixed price.

Team: {TEAM} · SRM Institute of Science and Technology, Delhi-NCR Campus
"""
    )
