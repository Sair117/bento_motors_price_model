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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional: Add custom CSS to clean up Streamlit's default look
st.markdown("""
<style>
    /* Clean up the main menu and footer for a more professional app feel */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Make metrics look sharper */
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    /* Simple styling for the predict button */
    .stButton>button {
        width: 100%;
        border-radius: 4px;
        height: 3rem;
    }
    /* Professional header styling */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #1f2937;
    }
</style>
""", unsafe_allow_html=True)

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
    st.error(f"Missing artifact file: {e}")
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
st.sidebar.title("Bento Motors")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Price Predictor", "Model Performance", "Model Interpretation", "About"]
)


# ═══════════════════════════════════════════════════════════════
# PAGE 1: Price Predictor
# ═══════════════════════════════════════════════════════════════
if page == "Price Predictor":
    st.title("Vehicle Valuation")
    st.markdown("Enter vehicle specifications below to generate an automated price prediction and SHAP analysis.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Vehicle Identity")
        make = st.selectbox("Make", artifacts['unique_makes'])
        available_models = artifacts['models_by_make'].get(make, ['Other'])
        model_name = st.selectbox("Model", available_models)
        year = st.slider("Year of Registration", 1990, 2020, 2017)

    with col2:
        st.subheader("Specifications")
        mileage = st.number_input("Mileage", 0, 500000, 30000, step=1000)
        fuel = st.selectbox("Fuel Type", artifacts['unique_fuels'])
        body_type = st.selectbox("Body Type", artifacts['unique_body_types'])

    with col3:
        st.subheader("Details")
        transmission = st.selectbox("Transmission",
                                     artifacts.get('unique_transmissions', ['Manual', 'Automatic']))
        condition = st.selectbox("Condition",
                                  artifacts.get('unique_conditions', ['USED', 'NEW']))

    st.markdown("---")

    if st.button("Generate Valuation", type="primary", use_container_width=True):
        input_df = preprocess_input(make, model_name, year, mileage, fuel,
                                     body_type, transmission, condition)

        predicted_price = max(best_model.predict(input_df)[0], 0)
        predicted_band = assign_band(predicted_price)

        st.markdown("---")
        r1, r2 = st.columns(2)
        r1.metric("Predicted Price", f"£{predicted_price:,.0f}")
        r2.metric("Market Segment", predicted_band)

        st.markdown("### Valuation Analysis")
        st.caption("SHAP feature attribution: Visualizing the impact of individual specifications on the final valuation.")

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
            st.info(f"Base Market Value: £{explainer.expected_value:,.0f}. "
                    f"Vehicle-specific features adjust the valuation to £{predicted_price:,.0f}.")
        except Exception as e:
            st.warning(f"SHAP explainer unavailable: {e}")


# ═══════════════════════════════════════════════════════════════
# PAGE 2: Model Performance
# ═══════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.title("Model Performance Dashboard")
    st.markdown(f"**Primary Model**: {best_model_name}")
    st.markdown("---")

    st.subheader("Regression Benchmarks")
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
        st.subheader("Actual vs Predicted (Validation)")
        if y_val is not None and val_preds is not None:
            mask = y_val < 100000
            fig = px.scatter(x=y_val[mask], y=val_preds[mask],
                            labels={'x': 'Actual Price (£)', 'y': 'Predicted Price (£)'},
                            opacity=0.15)
            fig.add_trace(go.Scatter(x=[0, 100000], y=[0, 100000],
                                     mode='lines', name='Perfect Alignment',
                                     line=dict(dash='dash', color='#ef4444')))
            fig.update_layout(height=400, showlegend=False, 
                              plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Residual Distribution (Test)")
        if y_test is not None and test_preds is not None:
            res = y_test - test_preds
            res_clip = res[(res > -30000) & (res < 30000)]
            fig = px.histogram(res_clip, nbins=100,
                              labels={'value': 'Valuation Error (£)', 'count': 'Frequency'},
                              color_discrete_sequence=['#3b82f6'])
            fig.update_layout(height=400, showlegend=False,
                              plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Prediction Precision")
    if test_preds is not None:
        abs_err = np.abs(y_test - test_preds)
        cols = st.columns(5)
        for i, p in enumerate([50, 75, 90, 95, 99]):
            cols[i].metric(f"P{p} Error Margin", f"£{np.percentile(abs_err, p):,.0f}")

    st.markdown("---")
    st.subheader("Price Segmentation Strategy")
    st.markdown("""
    | Market Segment | Price Range | Target Demographics / Inventory Profile |
    |----------------|-------------|---------------------------------------|
    | Budget | < £10K | Economy, reliable prior-owned inventory |
    | Mid-Range | £10K–£25K | Mainstream consumer vehicles |
    | Premium | £25K–£50K | Entry-to-mid level premium brands |
    | Luxury | £50K–£100K | Executive and luxury vehicles |
    | Supercar | > £100K | High-performance and exotic inventory |
    """)


# ═══════════════════════════════════════════════════════════════
# PAGE 3: Model Interpretation
# ═══════════════════════════════════════════════════════════════
elif page == "Model Interpretation":
    st.title("Model Interpretation")
    st.markdown(f"Interpretability analysis for **{best_model_name}**.")
    st.markdown("---")

    st.subheader("SHAP Summary (Global Feature Attribution)")
    st.caption("Distribution of feature influences across the validation cohort.")
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values_arr, X_val_sample, show=False, max_display=15)
        st.pyplot(fig)
        plt.close()
    except Exception as e:
        st.warning(f"SHAP summary analysis currently unavailable: {e}")

    st.markdown("---")

    st.subheader("Coefficient of Determination (R²) Benchmarks")
    results_raw = artifacts['results']
    comp_df = pd.DataFrame(results_raw).sort_values('R² Score', ascending=True)
    fig = px.bar(comp_df, x='R² Score', y='Model', orientation='h',
                 color='R² Score', color_continuous_scale='Blues',
                 text=comp_df['R² Score'].apply(lambda x: f"{x:.4f}"))
    fig.update_layout(height=400, showlegend=False, margin=dict(l=200),
                      plot_bgcolor='white', paper_bgcolor='white')
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Core Determinants of Value")
    st.markdown("""
    - **Make & Model (Target-Encoded)**: Represent the primary driving force behind vehicle valuation, reflecting brand equity.
    - **Mileage & Vehicle Age**: The definitive indicators of depreciation over time.
    - **Mileage per Year**: Acts as an engineered proxy for usage intensity and anticipated wear-and-tear.
    - **Body & Fuel Type**: Secondary determinants influencing market demand elasticity.
    """)

    st.markdown("---")
    st.subheader("Bias-Variance Trade-off Analysis")
    for r in sorted(results, key=lambda x: x['R² Score'], reverse=True):
        gap = r.get('Train R²', 0) - r['R² Score']
        if gap > 0.10: s = "Overfitting (High Variance)"
        elif gap > 0.05: s = "Slight Variance"
        elif r.get('Train R²', 0) < 0.70: s = "Underfitting (High Bias)"
        else: s = "Optimal Generalization"
        st.markdown(f"**{r['Model']}**: Training R²={r.get('Train R²',0):.4f}, "
                    f"Validation R²={r['R² Score']:.4f}, Delta={gap:.4f} — *{s}*")


# ═══════════════════════════════════════════════════════════════
# PAGE 4: About
# ═══════════════════════════════════════════════════════════════
elif page == "About":
    st.title("Project Overview")
    st.markdown("---")
    st.markdown("""
    ### Automated Vehicle Valuation Model

    A machine learning driven approach to used vehicle appraisal in the UK automotive market.

    #### Data Infrastructure
    - **Source**: Aggregated UK vehicle listings (~393,000 sanitized records)
    - **Variables**: Make, model, year, mileage, fuel type, body type, transmission, condition
    - **Target Variable**: Advertised Sale Price (£)

    #### Algorithmic Approach
    - **Regression Pipeline**: Utilizes CatBoost (R²=0.88 on mainstream segments representing 96% of market capitalization)
    - **Classification Strategy**: Employs Random Forest ensembles achieving 88.7% accuracy across 5 distinct market segments

    #### Preprocessing Methodology
    1. K-Fold target encoding for high-cardinality features (minimizing data leakage)
    2. Frequency distributions to capture category liquidity
    3. Scoped One-Hot encoding constrained to training distributions
    4. Derived feature integration (e.g., historical usage intensity via `mileage_per_year`)

    #### Explainable AI (XAI) Integration
    - Granular transparency achieved via SHapley Additive exPlanations (SHAP)
    - Supports both macro-level feature importance tracking and micro-level appraisal justification
    """)
    st.caption("Developed for enterprise-grade inventory valuation and analytics.")
