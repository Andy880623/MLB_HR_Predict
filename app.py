"""Streamlit app for MLB HR prediction project."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MLB 2025 HR Prediction", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SHAP_DIR = REPORTS_DIR / "shap_artifacts"

MODEL_FILES = {
    "Linear Regression": MODELS_DIR / "linear_regression.joblib",
    "Elastic Net (GridSearch)": MODELS_DIR / "elastic_net.joblib",
    "Decision Tree (GridSearch)": MODELS_DIR / "decision_tree.joblib",
    "Random Forest (GridSearch)": MODELS_DIR / "random_forest.joblib",
    "Extra Trees (RandomSearch)": MODELS_DIR / "extra_trees.joblib",
    "HistGradientBoosting (RandomSearch)": MODELS_DIR / "hist_gradient_boosting.joblib",
    "Stacking Ensemble": MODELS_DIR / "stacking_ensemble.joblib",
    "XGBoost (GridSearch)": MODELS_DIR / "xgboost.joblib",
    "LightGBM (GridSearch)": MODELS_DIR / "lightgbm.joblib",
    "Gradient Boosting (Fallback, GridSearch)": MODELS_DIR / "boosting_fallback.joblib",
    "MLP Regressor": MODELS_DIR / "mlp_regressor.joblib",
}


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path):
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource
def load_models():
    loaded = {}
    for model_name, model_path in MODEL_FILES.items():
        if model_path.exists():
            loaded[model_name] = joblib.load(model_path)
    return loaded


def is_tree_based_pipeline(model_obj) -> bool:
    estimator = model_obj.named_steps.get("model")
    tree_like_tokens = [
        "Tree",
        "Forest",
        "XGB",
        "LGBM",
        "GradientBoosting",
        "HistGradientBoosting",
    ]
    return any(token in estimator.__class__.__name__ for token in tree_like_tokens)


st.title("MLB 2025 Home Run Prediction")
st.caption("MSIS 522 end-to-end regression project")

metadata = load_json(MODELS_DIR / "metadata.json")
intro = load_json(REPORTS_DIR / "dataset_intro.json")
cleaning_summary = load_json(REPORTS_DIR / "data_cleaning_summary.json")
comparison_df = load_csv(REPORTS_DIR / "model_comparison.csv")
best_hyperparams = load_json(REPORTS_DIR / "best_hyperparameters.json")
pred_2025_df = load_csv(REPORTS_DIR / "predicted_2025_hr.csv")
models = load_models()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Executive Summary",
        "Descriptive Analytics",
        "Model Performance",
        "Explainability & Interactive Prediction",
    ]
)

with tab1:
    st.subheader("Project Overview")
    st.write(
        "Goal: Predict each player's 2025 home runs using player-season batting statistics "
        "and compare multiple regression models."
    )
    st.write(
        "This dataset captures year-by-year offensive performance across MLB players and lets us estimate "
        "future power output. The target is the following season's home run total, which makes this a practical "
        "forecasting problem for player valuation, roster planning, and fan-facing projections."
    )
    st.write(
        "The workflow includes data cleaning for traded-player rows, exploratory analysis, feature engineering, "
        "multiple model families, and explainability with SHAP. Results show that ensemble tree models perform "
        "best, balancing predictive strength with interpretable feature contributions."
    )

    if intro:
        c1, c2, c3 = st.columns(3)
        c1.metric("Raw Rows", intro["raw_shape"][0])
        c2.metric("Modeling Rows", intro["model_shape"][0])
        c3.metric("Players", intro["num_players"])
        st.write(f"Seasons covered: {intro['year_min']} to {intro['year_max']}")
        if "cleaned_shape" in intro:
            st.write(f"After removing 2TM and deduplicating player-year rows: {intro['cleaned_shape'][0]} rows")

    if cleaning_summary:
        st.info(
            f"Data cleaning applied: dropped {cleaning_summary['dropped_2tm_rows']} rows with Team='2TM', "
            f"and removed {cleaning_summary['dropped_duplicate_player_year_rows']} duplicate player-year rows."
        )

    if comparison_df is not None and not comparison_df.empty:
        best_row = comparison_df.sort_values("R2", ascending=False).iloc[0]
        st.success(
            f"Best overall model: {best_row['Model']} | "
            f"MAE={best_row['MAE']:.3f}, RMSE={best_row['RMSE']:.3f}, R2={best_row['R2']:.3f}"
        )

    if pred_2025_df is not None and not pred_2025_df.empty:
        st.subheader("Top Predicted HR Leaders for 2025")
        if "PA" in pred_2025_df.columns:
            pa_threshold = st.slider("Minimum PA filter for leaderboard", min_value=0, max_value=700, value=150, step=10)
            leader_df = pred_2025_df[pred_2025_df["PA"] >= pa_threshold].copy()
        else:
            leader_df = pred_2025_df.copy()
        leader_df = leader_df.sort_values("Predicted_HR_next_season", ascending=False)
        st.dataframe(leader_df.head(15), use_container_width=True)

        st.subheader("Power Leaderboard (PA >= 300 and Predicted HR >= 20)")
        col1, col2 = st.columns(2)
        power_pa_threshold = col1.slider(
            "Power leaderboard min PA", min_value=0, max_value=700, value=300, step=10
        )
        power_hr_threshold = col2.slider(
            "Power leaderboard min predicted HR", min_value=0.0, max_value=60.0, value=20.0, step=0.5
        )
        power_df = leader_df.copy()
        if "PA" in power_df.columns:
            power_df = power_df[power_df["PA"] >= power_pa_threshold]
        power_df = power_df[power_df["Predicted_HR_next_season"] >= power_hr_threshold]
        power_df = power_df.sort_values("Predicted_HR_next_season", ascending=False)

        if power_df.empty:
            st.warning("No players meet the current power leaderboard thresholds.")
        else:
            st.dataframe(power_df.head(20), use_container_width=True)

with tab2:
    st.subheader("EDA Visuals")
    plot_captions = {
        "Target Distribution": "This histogram shows that next-season HR is right-skewed, with many low-to-mid HR seasons and fewer extreme sluggers. This implies models can underpredict the long tail if not tuned carefully.",
        "PA vs Next-Season HR": "Players with higher plate appearances generally have a higher HR ceiling in the next season. The relationship is positive but noisy, which suggests volume helps but is not the only driver.",
        "Age Group Analysis": "Prime age bins show stronger HR outcomes relative to the oldest bin. This supports the use of age and non-linear age effects in modeling.",
        "Yearly Trends": "Average HR and OPS move together over time, indicating a changing offensive environment by season. Including year-related information helps account for era effects.",
        "Correlation Heatmap": "Several batting metrics are highly correlated, which can affect linear models and feature attribution. Tree ensembles usually handle these dependencies better than simple linear baselines.",
    }
    image_files = [
        ("Target Distribution", FIGURES_DIR / "01_target_distribution.png"),
        ("PA vs Next-Season HR", FIGURES_DIR / "02_pa_vs_target_scatter.png"),
        ("Age Group Analysis", FIGURES_DIR / "03_agebin_boxplot.png"),
        ("Yearly Trends", FIGURES_DIR / "04_yearly_trends.png"),
        ("Correlation Heatmap", FIGURES_DIR / "05_correlation_heatmap.png"),
    ]
    for title, img_path in image_files:
        if img_path.exists():
            st.markdown(f"**{title}**")
            st.image(str(img_path), use_container_width=True)
            st.caption(plot_captions.get(title, ""))

    notes_path = REPORTS_DIR / "eda_interpretations.txt"
    if notes_path.exists():
        st.subheader("Interpretations")
        st.text(notes_path.read_text(encoding="utf-8"))

with tab3:
    st.subheader("Model Comparison")
    if comparison_df is not None:
        st.dataframe(comparison_df, use_container_width=True)

    chart_path = FIGURES_DIR / "06_model_comparison_r2.png"
    if chart_path.exists():
        st.image(str(chart_path), use_container_width=True)

    st.markdown("**Evaluation Metrics**")
    st.write("- MAE: Average absolute prediction error")
    st.write("- RMSE: Penalizes larger errors more heavily")
    st.write("- R2: Proportion of variance explained (higher is better)")

    st.subheader("Best Hyperparameters by Model")
    if best_hyperparams:
        for model_name, params in best_hyperparams.items():
            with st.expander(model_name):
                st.json(params)
    else:
        st.info("best_hyperparameters.json not found yet. Rerun training to generate it.")

    st.subheader("Predicted vs Actual Plots (Test Set)")
    pred_actual_images = sorted(FIGURES_DIR.glob("pred_vs_actual_*.png"))
    if pred_actual_images:
        for img_path in pred_actual_images:
            title = img_path.stem.replace("pred_vs_actual_", "").replace("_", " ").title()
            st.markdown(f"**{title}**")
            st.image(str(img_path), use_container_width=True)
    else:
        st.info("Predicted-vs-Actual figures not found yet. Rerun training to generate them.")

with tab4:
    st.subheader("SHAP Explainability")
    shap_files = [
        ("SHAP Summary Plot", SHAP_DIR / "01_shap_summary.png"),
        ("SHAP Bar Plot", SHAP_DIR / "02_shap_bar.png"),
        ("SHAP Waterfall Plot", SHAP_DIR / "03_shap_waterfall.png"),
    ]
    for title, img_path in shap_files:
        if img_path.exists():
            st.markdown(f"**{title}**")
            st.image(str(img_path), use_container_width=True)

    st.subheader("Interactive Prediction")
    if not metadata or not models:
        st.warning("Train models first so metadata and model files are available.")
    else:
        selected_model_name = st.selectbox("Choose model", list(models.keys()))
        selected_model = models[selected_model_name]

        feature_columns = metadata["feature_columns"]
        num_defaults = metadata.get("numeric_defaults", {})
        cat_defaults = metadata.get("categorical_defaults", {})

        priority_num_features = [
            "Age",
            "WAR",
            "G",
            "PA",
            "AB",
            "R",
            "H",
            "HR",
            "RBI",
            "BB",
            "SO",
            "BA",
            "OBP",
            "SLG",
            "OPS",
            "Year",
        ]

        input_row = {}
        st.markdown("Adjust key features and click Predict:")

        for col in feature_columns:
            if col in num_defaults:
                if col in priority_num_features:
                    default_val = float(num_defaults[col])
                    input_row[col] = st.number_input(
                        f"{col}",
                        value=default_val,
                        format="%.4f",
                    )
                else:
                    input_row[col] = float(num_defaults[col])
            else:
                default_val = str(cat_defaults.get(col, "Unknown"))
                input_row[col] = st.text_input(f"{col}", value=default_val)

        if st.button("Predict Next-Season HR"):
            input_df = pd.DataFrame([input_row], columns=feature_columns)
            raw_pred = float(selected_model.predict(input_df)[0])
            use_log_target = bool(metadata.get("use_log_target", False))
            prediction = float(np.expm1(raw_pred)) if use_log_target else raw_pred
            prediction = max(prediction, 0.0)
            st.success(f"Predicted next-season HR: {prediction:.2f}")

            st.markdown("**SHAP Waterfall for This Custom Input**")
            if not is_tree_based_pipeline(selected_model):
                st.info("Selected model is not tree-based. Choose a tree-based model to view SHAP waterfall.")
            else:
                try:
                    import shap
                    from scipy import sparse

                    preprocessor = selected_model.named_steps["preprocessor"]
                    estimator = selected_model.named_steps["model"]

                    x_t = preprocessor.transform(input_df)
                    if sparse.issparse(x_t):
                        x_t = x_t.toarray()
                    feature_names = preprocessor.get_feature_names_out()
                    x_t_df = pd.DataFrame(x_t, columns=feature_names)

                    explainer = shap.TreeExplainer(estimator)
                    shap_values = explainer.shap_values(x_t_df)
                    expected_value = explainer.expected_value
                    if isinstance(expected_value, np.ndarray):
                        expected_value = float(expected_value[0])

                    exp = shap.Explanation(
                        values=shap_values[0],
                        base_values=expected_value,
                        data=x_t_df.iloc[0].values,
                        feature_names=x_t_df.columns.tolist(),
                    )
                    fig = plt.figure(figsize=(8, 5))
                    shap.plots.waterfall(exp, max_display=15, show=False)
                    st.pyplot(fig)
                    plt.close(fig)
                except ImportError:
                    st.warning("SHAP package is not installed in this environment.")
                except Exception as e:
                    st.warning(f"Unable to generate SHAP waterfall for this model/input: {e}")
