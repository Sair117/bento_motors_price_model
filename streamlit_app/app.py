"""
Car Price Prediction — Streamlit Dashboard
Model Deployment + Interpretation Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import shap
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from catboost import CatBoostRegressor

# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════
# Load Artifacts (2 files: model + app data)
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_all():
    base = Path(__file__).parent
    
    # CatBoost model (native format — ~2 MB)
    model = CatBoostRegressor()
    model.load_model(str(base / "best_model.cbm"))
    
    # App data (encodings, options, results — ~2 MB)
    with open(base / "app_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    return model, data

try:
    best_model, artifacts = load_all()
except FileNotFoundError as e:
    st.error(f"⚠️ Missing artifact file: {e}")
    st.info("Run the notebook's save cells and place best_model.cbm and app_data.pkl here.")
    st.stop()

# Unpack
best_model_name = artifacts['best_model_name']
feature_columns = artifacts['feature_columns']
make_te_map     = artifacts['make_te_map']
model_te_map    = artifacts['model_te_map']
make_freq_map   = artifacts['make_freq_map']
model_freq_map  = artifacts['model_freq_map']
global_mean     = artifacts['global_mean_price']
ohe             = artifacts['ohe']
ohe_cols        = artifacts['ohe_cols']
DATASET_YEAR    = artifacts['DATASET_YEAR']
results         = artifacts['results']
class_order     = artifacts['class_order']
shap_values_arr = artifacts['shap_values_sample']
shap_expected   = artifacts['shap_expected_value']
X_val_sample    = artifacts['X_shap_sample']
scaler          = artifacts['scaler']
scale_cols      = artifacts['scale_cols']


# ═══════════════════════════════════════════════════════════════
# Helper: Preprocess Input
# ═══════════════════════════════════════════════════════════════
def preprocess_input(make, model_name, year, mileage, fuel, body_type,
                     transmission, condition):
    car_age = DATASET_YEAR - year
    mileage_per_year = mileage / max(car_age, 1)
    mileage_per_year = min(mileage_per_year, 50000)

    features = {
        'year_of_registration': year,
        'mileage': mileage,
        'car_age': car_age,
        'mileage_per_year': mileage_per_year,
        'standard_make_encoded': make_te_map.get(make, global_mean),
        'standard_model_encoded': model_te_map.get(model_name, global_mean),
        'standard_make_freq': make_freq_map.get(make, 1),
        'standard_model_freq': model_freq_map.get(model_name, 1),
    }

    # OHE
    ohe_input = pd.DataFrame({
        'fuel_type': [fuel],
        'body_type': [body_type],
        'vehicle_condition': [condition],
        'standard_colour': ['White']
    })

    try:
        encoded = ohe.transform(ohe_input)
        ohe_names = ohe.get_feature_names_out(ohe_cols).tolist()
        for name, val in zip(ohe_names, encoded[0]):
            features[name] = int(val)
    except Exception:
        pass

    df = pd.DataFrame([features])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    colour_cols = [c for c in df.columns if 'standard_colour' in c]
    df = df.drop(columns=[c for c in colour_cols if c in df.columns], errors='ignore')
    df = df.reindex(columns=feature_columns, fill_value=0)
    
    # Apply the same scaling used during training
    cols_to_scale = [c for c in scale_cols if c in df.columns]
    if cols_to_scale:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    
    return df


def assign_band(price):
    if price < 10000: return 'Budget'
    elif price < 25000: return 'Mid-Range'
    elif price < 50000: return 'Premium'
    elif price < 100000: return 'Luxury'
    else: return 'Supercar'


# ═══════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════
st.sidebar.title("🚗 Car Price Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Price Predictor", "📊 Model Performance", "🔍 Model Interpretation", "ℹ️ About"]
)


# ═══════════════════════════════════════════════════════════════
# PAGE 1: Price Predictor
# ═══════════════════════════════════════════════════════════════
if page == "🏠 Price Predictor":
    st.title("🚗 Car Price Predictor")
    st.markdown("Enter car details below to get an instant price prediction with AI explanation.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🏷️ Vehicle Identity")
        make = st.selectbox("Make", artifacts['unique_makes'])
        available_models = artifacts['models_by_make'].get(make, ['Other'])
        model_name = st.selectbox("Model", available_models)
        year = st.slider("Year of Registration", 1990, 2020, 2017)

    with col2:
        st.subheader("📋 Specifications")
        mileage = st.number_input("Mileage", 0, 500000, 30000, step=1000)
        fuel = st.selectbox("Fuel Type", artifacts['unique_fuels'])
        body_type = st.selectbox("Body Type", artifacts['unique_body_types'])

    with col3:
        st.subheader("⚙️ Details")
        transmission = st.selectbox("Transmission",
                                     artifacts.get('unique_transmissions', ['Manual', 'Automatic']))
        condition = st.selectbox("Condition",
                                  artifacts.get('unique_conditions', ['USED', 'NEW']))

    st.markdown("---")

    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        input_df = preprocess_input(make, model_name, year, mileage, fuel,
                                     body_type, transmission, condition)

        predicted_price = max(best_model.predict(input_df)[0], 0)
        predicted_band = assign_band(predicted_price)

        st.markdown("---")
        r1, r2 = st.columns(2)
        r1.metric("💰 Predicted Price", f"£{predicted_price:,.0f}")
        r2.metric("📊 Price Band", predicted_band)

        st.markdown("### 🧠 Why This Price?")
        st.caption("SHAP values show how each feature pushes the prediction up or down from the average.")

        try:
            explainer = shap.TreeExplainer(best_model)
            sv = explainer.shap_values(input_df)
            explanation = shap.Explanation(
                values=sv[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0].values,
                feature_names=feature_columns
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(explanation, max_display=12, show=False)
            st.pyplot(fig)
            plt.close()
            st.info(f"**Base value**: £{explainer.expected_value:,.0f} (average). "
                    f"Features push prediction to **£{predicted_price:,.0f}**.")
        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")


# ═══════════════════════════════════════════════════════════════
# PAGE 2: Model Performance
# ═══════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance Dashboard")
    st.markdown(f"**Best model**: {best_model_name}")
    st.markdown("---")

    st.subheader("🏆 Model Comparison")
    rdf = pd.DataFrame(results).sort_values('R² Score', ascending=False)
    display_df = rdf.copy()
    display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f"£{x:,.0f}")
    display_df['MAE'] = display_df['MAE'].apply(lambda x: f"£{x:,.0f}")
    display_df['R² Score'] = display_df['R² Score'].apply(lambda x: f"{x:.4f}")
    display_df['Train R²'] = display_df['Train R²'].apply(lambda x: f"{x:.4f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    y_val = artifacts.get('y_val')
    val_preds = artifacts.get('val_preds')
    y_test = artifacts.get('y_test')
    test_preds = artifacts.get('test_preds')

    with col1:
        st.subheader("📈 Actual vs Predicted (Validation)")
        if y_val is not None and val_preds is not None:
            mask = y_val < 100000
            fig = px.scatter(x=y_val[mask], y=val_preds[mask],
                            labels={'x': 'Actual (£)', 'y': 'Predicted (£)'},
                            opacity=0.15)
            fig.add_trace(go.Scatter(x=[0, 100000], y=[0, 100000],
                                     mode='lines', name='Perfect',
                                     line=dict(dash='dash', color='red')))
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📉 Residual Distribution (Test)")
        if y_test is not None and test_preds is not None:
            res = y_test - test_preds
            res_clip = res[(res > -30000) & (res < 30000)]
            fig = px.histogram(res_clip, nbins=100,
                              labels={'value': 'Error (£)', 'count': 'Count'})
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Prediction Accuracy")
    if test_preds is not None:
        abs_err = np.abs(y_test - test_preds)
        cols = st.columns(5)
        for i, p in enumerate([50, 75, 90, 95, 99]):
            cols[i].metric(f"{p}th Percentile Error", f"£{np.percentile(abs_err, p):,.0f}")

    st.markdown("---")
    st.subheader("🏷️ 5-Band Classification")
    st.markdown("""
    | Band | Range | Description |
    |------|-------|-------------|
    | Budget | < £10K | Economy / older used cars |
    | Mid-Range | £10K–£25K | Mainstream family cars |
    | Premium | £25K–£50K | New premium brands |
    | Luxury | £50K–£100K | Luxury daily drivers |
    | Supercar | > £100K | Exotic / performance |
    """)


# ═══════════════════════════════════════════════════════════════
# PAGE 3: Model Interpretation
# ═══════════════════════════════════════════════════════════════
elif page == "🔍 Model Interpretation":
    st.title("🔍 Model Interpretation")
    st.markdown(f"Understanding what **{best_model_name}** has learned.")
    st.markdown("---")

    st.subheader("🌊 SHAP Summary — Global Feature Impact")
    st.caption("Each dot = one car. Red = high value, blue = low. Position = impact on price.")
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values_arr, X_val_sample, show=False, max_display=15)
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"SHAP summary unavailable: {e}")

    st.markdown("---")

    st.subheader("📊 Feature Importance")
    importance = best_model.feature_importances_
    feat_imp = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importance
    }).sort_values('Importance', ascending=True).tail(15)
    fig = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
                 color='Importance', color_continuous_scale='Blues')
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("💡 Key Insights")
    st.markdown("""
    - **Make & Model** (target-encoded) dominate — brand is the #1 price driver
    - **Mileage** and **Car Age** capture depreciation
    - **Mileage per Year** reflects usage intensity
    - **Body Type** and **Fuel Type** have moderate effects
    """)

    st.markdown("---")
    st.subheader("⚖️ Bias-Variance Analysis")
    for r in sorted(results, key=lambda x: x['R² Score'], reverse=True):
        gap = r.get('Train R²', 0) - r['R² Score']
        if gap > 0.10: s = "🔴 Overfitting"
        elif gap > 0.05: s = "🟡 Slight overfit"
        elif r.get('Train R²', 0) < 0.70: s = "🔵 Underfitting"
        else: s = "🟢 Balanced"
        st.markdown(f"**{r['Model']}**: Train={r.get('Train R²',0):.4f}, "
                    f"Val={r['R² Score']:.4f}, Gap={gap:.4f} — {s}")


# ═══════════════════════════════════════════════════════════════
# PAGE 4: About
# ═══════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("---")
    st.markdown("""
    ### 🚗 Car Price Prediction Model

    ML-powered used car price prediction for the UK market.

    #### Dataset
    - **Source**: UK used car listings (~393,000 records)
    - **Features**: Make, model, year, mileage, fuel, body type, transmission, condition
    - **Target**: Sale price (£)

    #### Models
    - **Regression**: CatBoost — R²=0.88 on mainstream cars (<£50K, 96% of market)
    - **Classification**: Random Forest — 88.7% accuracy across 5 price bands

    #### Pipeline
    1. K-fold target encoding (prevents leakage)
    2. Frequency encoding (category prevalence)
    3. OneHot encoding (fitted on training data only)
    4. Feature engineering: car_age, mileage_per_year

    #### Interpretation
    - SHAP provides full transparency for every prediction
    - Global and individual-level explanations available
    """)
    st.caption("Built as part of an ML coursework project.")
