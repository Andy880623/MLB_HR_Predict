"""End-to-end MLB HR prediction project (MSIS 522).

This script:
1) Loads and validates batting data.
2) Cleans traded-team duplicates (drop 2TM and keep one row per player-year).
3) Builds next-season HR target with temporal features.
4) Runs EDA with required visualizations.
5) Trains and compares five regression models.
6) Performs SHAP analysis on the best tree-based model.
7) Saves trained models and project artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib

# Keep matplotlib cache writable in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp/matplotlib_cache").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path("tmp").resolve()))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")

RANDOM_STATE = 42
TARGET_COL = "HR_next_season"
USE_LOG_TARGET = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLB HR prediction models.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/cleaned_batting_stats.csv",
        help="Path to cleaned_batting_stats.csv",
    )
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [
        "data/processed",
        "models",
        "reports/figures",
        "reports/shap_artifacts",
        "tmp/matplotlib_cache",
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    required_cols = {"Player", "Year", "HR"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def clean_player_year_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop 2TM rows and keep a single representative row per player-year.

    Strategy:
    - Remove rows where Team == 2TM.
    - If a player still has multiple team rows in a year, keep the row with largest PA.
    """
    data = df.copy()

    data["Team"] = data["Team"].astype(str)
    before_rows = len(data)
    data = data[data["Team"] != "2TM"].copy()
    after_drop_2tm = len(data)

    if "PA" in data.columns:
        data["PA"] = pd.to_numeric(data["PA"], errors="coerce").fillna(0)
        data = data.sort_values(["Player", "Year", "PA"], ascending=[True, True, False])
        data = data.drop_duplicates(subset=["Player", "Year"], keep="first")
    else:
        data = data.sort_values(["Player", "Year"]).drop_duplicates(subset=["Player", "Year"], keep="first")

    data = data.sort_values(["Player", "Year"]).reset_index(drop=True)

    with open("reports/data_cleaning_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "rows_before": before_rows,
                "rows_after_drop_2tm": after_drop_2tm,
                "rows_after_player_year_dedup": int(len(data)),
                "dropped_2tm_rows": int(before_rows - after_drop_2tm),
                "dropped_duplicate_player_year_rows": int(after_drop_2tm - len(data)),
            },
            f,
            indent=2,
        )

    return data


def add_temporal_features(player_year_df: pd.DataFrame) -> pd.DataFrame:
    """Add lag and rolling features by player to improve temporal signal."""
    data = player_year_df.copy()
    g = data.groupby("Player")

    data["HR_lag1"] = g["HR"].shift(1)
    data["HR_lag2"] = g["HR"].shift(2)
    if "PA" in data.columns:
        data["PA_lag1"] = g["PA"].shift(1)
    if "OPS" in data.columns:
        data["OPS_lag1"] = g["OPS"].shift(1)

    if "PA" in data.columns:
        denom = data["PA"].replace(0, np.nan)
        data["HR_per_PA"] = data["HR"] / denom

    data["HR_roll3_mean"] = (
        g["HR"].shift(1).rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    if "Age" in data.columns:
        data["Age_sq"] = pd.to_numeric(data["Age"], errors="coerce") ** 2

    return data


def build_supervised_dataset(player_year_df: pd.DataFrame) -> pd.DataFrame:
    """Create next-season HR target for each player-season row."""
    data = player_year_df.copy()
    data = data.sort_values(["Player", "Year"]).reset_index(drop=True)
    data[TARGET_COL] = data.groupby("Player")["HR"].shift(-1)

    model_data = data.dropna(subset=[TARGET_COL]).copy()
    model_data[TARGET_COL] = pd.to_numeric(model_data[TARGET_COL], errors="coerce")
    model_data = model_data.dropna(subset=[TARGET_COL])

    drop_cols = ["Rk", "RowNum", "Awards"]
    existing_drop_cols = [c for c in drop_cols if c in model_data.columns]
    model_data = model_data.drop(columns=existing_drop_cols)

    return model_data


def save_dataset_intro(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame, model_data: pd.DataFrame) -> None:
    years = sorted(raw_df["Year"].dropna().astype(int).unique().tolist())
    intro = {
        "raw_shape": raw_df.shape,
        "cleaned_shape": cleaned_df.shape,
        "model_shape": model_data.shape,
        "year_min": int(min(years)),
        "year_max": int(max(years)),
        "num_players": int(cleaned_df["Player"].nunique()),
        "target_description": "HR_next_season is each player's HR in the following season.",
    }
    with open("reports/dataset_intro.json", "w", encoding="utf-8") as f:
        json.dump(intro, f, indent=2)


def make_eda_plots(model_data: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(model_data[TARGET_COL], kde=True, bins=30, color="#1f77b4")
    plt.title("Target Distribution: Next-Season Home Runs")
    plt.xlabel(TARGET_COL)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("reports/figures/01_target_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=model_data, x="PA", y=TARGET_COL, alpha=0.5)
    sns.regplot(data=model_data, x="PA", y=TARGET_COL, scatter=False, color="red")
    plt.title("Plate Appearances vs Next-Season HR")
    plt.tight_layout()
    plt.savefig("reports/figures/02_pa_vs_target_scatter.png", dpi=200)
    plt.close()

    age_data = model_data.copy()
    age_data["AgeBin"] = pd.cut(
        age_data["Age"],
        bins=[15, 22, 26, 30, 34, 45],
        labels=["16-22", "23-26", "27-30", "31-34", "35-45"],
    )
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=age_data, x="AgeBin", y=TARGET_COL)
    plt.title("Next-Season HR by Age Group")
    plt.xlabel("Age Group")
    plt.tight_layout()
    plt.savefig("reports/figures/03_agebin_boxplot.png", dpi=200)
    plt.close()

    yearly = (
        model_data.groupby("Year")[["HR", TARGET_COL, "OPS"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    fig, ax1 = plt.subplots(figsize=(9, 5))
    sns.lineplot(data=yearly, x="Year", y="HR", marker="o", label="Current HR", ax=ax1)
    sns.lineplot(data=yearly, x="Year", y=TARGET_COL, marker="o", label="Next HR", ax=ax1)
    ax1.set_ylabel("Average HR")

    ax2 = ax1.twinx()
    sns.lineplot(data=yearly, x="Year", y="OPS", marker="s", color="green", label="OPS", ax=ax2)
    ax2.set_ylabel("Average OPS")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    plt.title("Yearly Batting Trends")
    plt.tight_layout()
    plt.savefig("reports/figures/04_yearly_trends.png", dpi=200)
    plt.close()

    numeric_cols = model_data.select_dtypes(include=[np.number]).columns.tolist()
    corr_cols = [c for c in numeric_cols if c not in ["Year"]]
    corr = model_data[corr_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=False)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig("reports/figures/05_correlation_heatmap.png", dpi=200)
    plt.close()

    notes = [
        "Target distribution is right-skewed, with many low-HR seasons and fewer power outliers.",
        "Higher plate appearances generally align with higher next-season HR potential.",
        "Prime-age bins tend to show stronger HR distributions than the oldest bin.",
        "Yearly HR and OPS move together, indicating offensive environment effects.",
        "Correlation heatmap highlights strong collinearity among rate and counting stats.",
    ]
    with open("reports/eda_interpretations.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(notes))


def time_based_split(X: pd.DataFrame, y_model: pd.Series, y_raw: pd.Series) -> Tuple[pd.DataFrame, ...]:
    """Split by year to simulate forecasting rather than random shuffle."""
    years = sorted(X["Year"].dropna().astype(int).unique().tolist())
    if len(years) < 3:
        raise ValueError("Need at least 3 distinct years for time-based split.")

    test_year = years[-1]
    train_mask = X["Year"].astype(int) < test_year
    test_mask = X["Year"].astype(int) == test_year

    X_train = X.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()

    y_train_model = y_model.loc[train_mask].copy()
    y_test_model = y_model.loc[test_mask].copy()
    y_train_raw = y_raw.loc[train_mask].copy()
    y_test_raw = y_raw.loc[test_mask].copy()

    return X_train, X_test, y_train_model, y_test_model, y_train_raw, y_test_raw


def build_year_cv_splits(X_train: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Rolling year-based CV splits inside train period."""
    years = sorted(X_train["Year"].dropna().astype(int).unique().tolist())
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    year_values = X_train["Year"].astype(int).to_numpy()
    for i in range(1, len(years)):
        train_years = years[:i]
        val_year = years[i]
        train_idx = np.where(np.isin(year_values, train_years))[0]
        val_idx = np.where(year_values == val_year)[0]
        if len(train_idx) > 0 and len(val_idx) > 0:
            splits.append((train_idx, val_idx))
    # Keep runtime controlled while preserving temporal order.
    return splits[-3:] if len(splits) > 3 else splits


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def build_numeric_only_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, numeric_features)],
        remainder="drop",
    )


def predict_on_raw_scale(model: Pipeline, X: pd.DataFrame, use_log_target: bool) -> np.ndarray:
    preds = model.predict(X)
    if use_log_target:
        preds = np.expm1(preds)
    return np.clip(preds, 0, None)


def evaluate_model(
    name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test_raw: pd.Series,
    use_log_target: bool,
) -> Dict:
    preds = predict_on_raw_scale(model, X_test, use_log_target)
    mae = mean_absolute_error(y_test_raw, preds)
    rmse = np.sqrt(mean_squared_error(y_test_raw, preds))
    r2 = r2_score(y_test_raw, preds)
    return {
        "Model": name,
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
    }


def _clean_param_keys(params: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for k, v in params.items():
        nk = k.replace("model__", "")
        if hasattr(v, "item"):
            try:
                v = v.item()
            except Exception:
                pass
        cleaned[nk] = v
    return cleaned


def train_models(
    X_train: pd.DataFrame, y_train_model: pd.Series, preprocessor: ColumnTransformer
) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, Any]]]:
    models = {}
    best_params: Dict[str, Dict[str, Any]] = {}
    cv_splits = 3
    print("CV mode: k-fold=3")

    linear = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    linear.fit(X_train, y_train_model)
    models["Linear Regression"] = linear
    best_params["Linear Regression"] = {"fit_intercept": True}
    print("Trained: Linear Regression")

    elastic = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", ElasticNet(random_state=RANDOM_STATE, max_iter=10000)),
        ]
    )
    elastic_grid = {
        "model__alpha": [0.001, 0.01, 0.1, 1.0],
        "model__l1_ratio": [0.2, 0.5, 0.8],
    }
    elastic_search = GridSearchCV(
        elastic,
        elastic_grid,
        cv=cv_splits,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    elastic_search.fit(X_train, y_train_model)
    models["Elastic Net (GridSearch)"] = elastic_search.best_estimator_
    best_params["Elastic Net (GridSearch)"] = _clean_param_keys(elastic_search.best_params_)
    print("Trained: Elastic Net")

    dt_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", DecisionTreeRegressor(random_state=RANDOM_STATE)),
        ]
    )
    dt_grid = {
        "model__max_depth": [8, None],
        "model__min_samples_split": [2, 10],
        "model__min_samples_leaf": [1, 4],
    }
    dt_search = GridSearchCV(dt_pipeline, dt_grid, cv=cv_splits, scoring="neg_mean_squared_error", n_jobs=1)
    dt_search.fit(X_train, y_train_model)
    models["Decision Tree (GridSearch)"] = dt_search.best_estimator_
    best_params["Decision Tree (GridSearch)"] = _clean_param_keys(dt_search.best_params_)
    print("Trained: Decision Tree")

    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)),
        ]
    )
    rf_grid = {
        "model__n_estimators": [200, 300],
        "model__max_depth": [None, 12, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.6, 0.8],
    }
    rf_search = RandomizedSearchCV(
        rf_pipeline,
        rf_grid,
        n_iter=4,
        random_state=RANDOM_STATE,
        cv=cv_splits,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    rf_search.fit(X_train, y_train_model)
    models["Random Forest (GridSearch)"] = rf_search.best_estimator_
    best_params["Random Forest (GridSearch)"] = _clean_param_keys(rf_search.best_params_)
    print("Trained: Random Forest")

    et_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", ExtraTreesRegressor(random_state=RANDOM_STATE, n_jobs=1)),
        ]
    )
    et_grid = {
        "model__n_estimators": [200, 300],
        "model__max_depth": [None, 12, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.6, 0.8],
    }
    et_search = RandomizedSearchCV(
        et_pipeline,
        et_grid,
        n_iter=4,
        random_state=RANDOM_STATE,
        cv=cv_splits,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    et_search.fit(X_train, y_train_model)
    models["Extra Trees (RandomSearch)"] = et_search.best_estimator_
    best_params["Extra Trees (RandomSearch)"] = _clean_param_keys(et_search.best_params_)
    print("Trained: Extra Trees")

    boosting_name = None
    try:
        from xgboost import XGBRegressor

        boost_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBRegressor(
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                        n_estimators=350,
                        n_jobs=1,
                    ),
                ),
            ]
        )
        boost_grid = {
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.7, 1.0],
        }
        boosting_name = "XGBoost (GridSearch)"
    except ImportError:
        try:
            from lightgbm import LGBMRegressor

            boost_pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        LGBMRegressor(random_state=RANDOM_STATE, n_estimators=350, verbosity=-1),
                    ),
                ]
            )
            boost_grid = {
                "model__max_depth": [-1, 8, 12],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__subsample": [0.8, 1.0],
            }
            boosting_name = "LightGBM (GridSearch)"
        except ImportError:
            boost_pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
                ]
            )
            boost_grid = {
                "model__n_estimators": [220, 320, 420],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [2, 3, 4],
                "model__subsample": [0.8, 1.0],
                "model__min_samples_leaf": [1, 3, 5],
            }
            boosting_name = "Gradient Boosting (Fallback, GridSearch)"
            print("Warning: xgboost/lightgbm not installed. Using GradientBoostingRegressor fallback.")

    boost_search = RandomizedSearchCV(
        boost_pipeline,
        boost_grid,
        n_iter=5,
        random_state=RANDOM_STATE,
        cv=cv_splits,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    boost_search.fit(X_train, y_train_model)
    models[boosting_name] = boost_search.best_estimator_
    best_params[boosting_name] = _clean_param_keys(boost_search.best_params_)
    print(f"Trained: {boosting_name}")

    hist_gb = Pipeline(
        steps=[
            ("preprocessor", build_numeric_only_preprocessor(X_train)),
            ("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
        ]
    )
    hist_gb_grid = {
        "model__learning_rate": [0.03, 0.05, 0.1],
        "model__max_depth": [None, 6, 10],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [10, 20, 40],
        "model__l2_regularization": [0.0, 0.01, 0.1],
    }
    hist_search = RandomizedSearchCV(
        hist_gb,
        hist_gb_grid,
        n_iter=5,
        random_state=RANDOM_STATE,
        cv=cv_splits,
        scoring="neg_mean_squared_error",
        n_jobs=1,
    )
    hist_search.fit(X_train, y_train_model)
    models["HistGradientBoosting (RandomSearch)"] = hist_search.best_estimator_
    best_params["HistGradientBoosting (RandomSearch)"] = _clean_param_keys(hist_search.best_params_)
    print("Trained: HistGradientBoosting")

    stack_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                StackingRegressor(
                    estimators=[
                        (
                            "rf",
                            RandomForestRegressor(
                                n_estimators=300,
                                max_features="sqrt",
                                random_state=RANDOM_STATE,
                                n_jobs=1,
                            ),
                        ),
                        (
                            "et",
                            ExtraTreesRegressor(
                                n_estimators=300,
                                max_features="sqrt",
                                random_state=RANDOM_STATE,
                                n_jobs=1,
                            ),
                        ),
                        (
                            "gb",
                            GradientBoostingRegressor(
                                n_estimators=320,
                                learning_rate=0.05,
                                max_depth=3,
                                subsample=0.8,
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ],
                    final_estimator=RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0]),
                    cv=3,
                    n_jobs=1,
                ),
            ),
        ]
    )
    stack_pipeline.fit(X_train, y_train_model)
    models["Stacking Ensemble"] = stack_pipeline
    best_params["Stacking Ensemble"] = {
        "base_estimators": ["RandomForestRegressor", "ExtraTreesRegressor", "GradientBoostingRegressor"],
        "final_estimator": "RidgeCV",
        "cv": 3,
    }
    print("Trained: Stacking Ensemble")

    mlp = Pipeline(
        steps=[
            ("preprocessor", build_numeric_only_preprocessor(X_train)),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    alpha=1e-3,
                    learning_rate_init=0.0005,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=25,
                    max_iter=300,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    mlp.fit(X_train, y_train_model)
    models["MLP Regressor"] = mlp
    best_params["MLP Regressor"] = {
        "hidden_layer_sizes": [128, 64],
        "activation": "relu",
        "alpha": 1e-3,
        "learning_rate_init": 5e-4,
        "early_stopping": True,
        "max_iter": 300,
    }
    print("Trained: MLP")

    return models, best_params


def save_model_metrics(results_df: pd.DataFrame) -> None:
    results_df.to_csv("reports/model_comparison.csv", index=False)

    plt.figure(figsize=(10, 5))
    plot_df = results_df.sort_values("R2", ascending=False)
    sns.barplot(data=plot_df, x="Model", y="R2", hue="Model", palette="viridis", legend=False)
    plt.title("Model Comparison (R2 on Test Set)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig("reports/figures/06_model_comparison_r2.png", dpi=200)
    plt.close()


def save_predicted_vs_actual_plots(
    models: Dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test_raw: pd.Series,
    use_log_target: bool,
) -> None:
    for model_name, model_obj in models.items():
        preds = predict_on_raw_scale(model_obj, X_test, use_log_target)
        plot_df = pd.DataFrame({"Actual": y_test_raw, "Predicted": preds})

        plt.figure(figsize=(6.8, 6))
        sns.scatterplot(data=plot_df, x="Actual", y="Predicted", alpha=0.45, s=28)
        min_v = float(min(plot_df["Actual"].min(), plot_df["Predicted"].min()))
        max_v = float(max(plot_df["Actual"].max(), plot_df["Predicted"].max()))
        plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="red", linewidth=1.5)
        plt.title(f"Predicted vs Actual: {model_name}")
        plt.xlabel("Actual HR_next_season")
        plt.ylabel("Predicted HR_next_season")
        plt.tight_layout()
        safe_name = (
            model_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace(",", "")
        )
        plt.savefig(f"reports/figures/pred_vs_actual_{safe_name}.png", dpi=200)
        plt.close()


def save_best_hyperparameters(best_params: Dict[str, Dict[str, Any]]) -> None:
    with open("reports/best_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)


def choose_best_tree_model(results_df: pd.DataFrame, model_store: Dict[str, Pipeline]) -> Tuple[str, Pipeline]:
    tree_names = [
        name
        for name in results_df["Model"].tolist()
        if any(
            token in name
            for token in [
                "Decision Tree",
                "Random Forest",
                "Extra Trees",
                "HistGradientBoosting",
                "XGBoost",
                "LightGBM",
                "Gradient Boosting",
            ]
        )
    ]
    tree_df = results_df[results_df["Model"].isin(tree_names)].copy()
    best_name = tree_df.sort_values("R2", ascending=False).iloc[0]["Model"]
    return best_name, model_store[best_name]


def run_shap(
    best_tree_name: str,
    best_tree_model: Pipeline,
    X_test: pd.DataFrame,
    sample_idx: int = 0,
) -> None:
    try:
        import shap
    except ImportError:
        with open("reports/shap_artifacts/README.txt", "w", encoding="utf-8") as f:
            f.write(
                "SHAP analysis skipped because shap is not installed. "
                "Install dependencies from requirements.txt and rerun training."
            )
        print("Warning: shap not installed. Skipping SHAP artifact generation.")
        return

    preprocessor = best_tree_model.named_steps["preprocessor"]
    estimator = best_tree_model.named_steps["model"]

    X_t = preprocessor.transform(X_test)
    if sparse.issparse(X_t):
        X_t = X_t.toarray()

    feature_names = preprocessor.get_feature_names_out()
    X_t_df = pd.DataFrame(X_t, columns=feature_names)

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_t_df)

    plt.figure()
    shap.summary_plot(shap_values, X_t_df, show=False, max_display=20)
    plt.title(f"SHAP Summary: {best_tree_name}")
    plt.tight_layout()
    plt.savefig("reports/shap_artifacts/01_shap_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_t_df, plot_type="bar", show=False, max_display=20)
    plt.title(f"SHAP Feature Importance (Bar): {best_tree_name}")
    plt.tight_layout()
    plt.savefig("reports/shap_artifacts/02_shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    expected_value = explainer.expected_value
    if isinstance(expected_value, np.ndarray):
        expected_value = float(expected_value[0])

    exp = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=expected_value,
        data=X_t_df.iloc[sample_idx].values,
        feature_names=X_t_df.columns.tolist(),
    )

    plt.figure()
    shap.plots.waterfall(exp, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig("reports/shap_artifacts/03_shap_waterfall.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_models(models: Dict[str, Pipeline]) -> None:
    model_name_map = {
        "Linear Regression": "linear_regression.joblib",
        "Elastic Net (GridSearch)": "elastic_net.joblib",
        "Decision Tree (GridSearch)": "decision_tree.joblib",
        "Random Forest (GridSearch)": "random_forest.joblib",
        "Extra Trees (RandomSearch)": "extra_trees.joblib",
        "HistGradientBoosting (RandomSearch)": "hist_gradient_boosting.joblib",
        "Stacking Ensemble": "stacking_ensemble.joblib",
        "XGBoost (GridSearch)": "xgboost.joblib",
        "LightGBM (GridSearch)": "lightgbm.joblib",
        "Gradient Boosting (Fallback, GridSearch)": "boosting_fallback.joblib",
        "MLP Regressor": "mlp_regressor.joblib",
    }
    for model_name, model_obj in models.items():
        file_name = model_name_map[model_name]
        joblib.dump(model_obj, Path("models") / file_name)


def save_metadata(
    cleaned_df: pd.DataFrame,
    X_train: pd.DataFrame,
    feature_columns: List[str],
    best_tree_name: str,
    results_df: pd.DataFrame,
) -> None:
    numeric_defaults = {}
    categorical_defaults = {}

    for col in X_train.columns:
        if pd.api.types.is_numeric_dtype(X_train[col]):
            numeric_defaults[col] = float(X_train[col].median())
        else:
            mode_vals = X_train[col].mode(dropna=True)
            categorical_defaults[col] = str(mode_vals.iloc[0]) if not mode_vals.empty else "Unknown"

    prediction_base_year = int(cleaned_df["Year"].max())

    metadata = {
        "target_col": TARGET_COL,
        "feature_columns": feature_columns,
        "numeric_defaults": numeric_defaults,
        "categorical_defaults": categorical_defaults,
        "best_tree_model": best_tree_name,
        "model_ranking": results_df.sort_values("R2", ascending=False)["Model"].tolist(),
        "prediction_base_year": prediction_base_year,
        "prediction_target_year": prediction_base_year + 1,
        "use_log_target": USE_LOG_TARGET,
    }

    with open("models/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def save_2025_predictions(
    latest_feature_df: pd.DataFrame,
    models: Dict[str, Pipeline],
    best_model_name: str,
    feature_columns: List[str],
) -> None:
    """Predict next-season HR using latest available season stats (e.g., 2024 -> 2025)."""
    base_year = int(latest_feature_df["Year"].max())
    pred_df = latest_feature_df[latest_feature_df["Year"] == base_year].copy()

    X_pred = pred_df[feature_columns].copy()
    best_model = models[best_model_name]
    pred_df["Predicted_HR_next_season"] = predict_on_raw_scale(best_model, X_pred, USE_LOG_TARGET)
    pred_df["Predicted_Season"] = base_year + 1

    # Safety dedup for leaderboard display.
    pred_df = pred_df.sort_values(["Player", "PA"], ascending=[True, False]).drop_duplicates("Player", keep="first")

    keep_cols = [
        c
        for c in [
            "Player",
            "Team",
            "Year",
            "Predicted_Season",
            "PA",
            "HR",
            "Predicted_HR_next_season",
        ]
        if c in pred_df.columns
    ]
    output = pred_df[keep_cols].sort_values("Predicted_HR_next_season", ascending=False)
    output.to_csv("reports/predicted_2025_hr.csv", index=False)


def main() -> None:
    args = parse_args()
    ensure_dirs()

    raw_df = load_data(args.data_path)
    cleaned_df = clean_player_year_rows(raw_df)
    engineered_df = add_temporal_features(cleaned_df)
    model_data = build_supervised_dataset(engineered_df)

    save_dataset_intro(raw_df, cleaned_df, model_data)
    make_eda_plots(model_data)

    excluded_features = [col for col in ["Player"] if col in model_data.columns]
    X = model_data.drop(columns=[TARGET_COL] + excluded_features)
    y_raw = model_data[TARGET_COL]
    y_model = np.log1p(y_raw) if USE_LOG_TARGET else y_raw

    X_train, X_test, y_train_model, y_test_model, y_train_raw, y_test_raw = train_test_split(
        X,
        y_model,
        y_raw,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    preprocessor, _, _ = build_preprocessor(X_train)
    models, best_params = train_models(X_train, y_train_model, preprocessor)

    results = []
    for name, model in models.items():
        results.append(evaluate_model(name, model, X_test, y_test_raw, USE_LOG_TARGET))
    results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)

    save_model_metrics(results_df)
    save_predicted_vs_actual_plots(models, X_test, y_test_raw, USE_LOG_TARGET)
    save_best_hyperparameters(best_params)

    best_tree_name, best_tree_model = choose_best_tree_model(results_df, models)
    run_shap(best_tree_name, best_tree_model, X_test, sample_idx=0)

    save_models(models)
    save_metadata(cleaned_df, X_train, X.columns.tolist(), best_tree_name, results_df)
    save_2025_predictions(engineered_df, models, best_tree_name, X.columns.tolist())

    model_data.to_csv("data/processed/modeling_dataset.csv", index=False)
    print("Training and analysis completed successfully.")
    print("Best tree-based model:", best_tree_name)
    print("Model comparison saved to reports/model_comparison.csv")


if __name__ == "__main__":
    main()
