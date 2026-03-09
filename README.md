# MLB 2025 HR Prediction (MSIS 522)

End-to-end regression project using the Kaggle dataset **MLB Hitting Data (2015–2024)** by `josuefernandezc`.

## Objective
Predict **2025 season home runs (HR)** from player-season batting statistics.

## Project Structure

```
MLB_HR_Predict/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
│   ├── figures/
│   └── shap_artifacts/
└── src/
    └── train.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data
Place `cleaned_batting_stats.csv` into `data/raw/` or pass its full path with `--data-path`.

Expected key columns include: `Player`, `Year`, `HR` (plus batting features).

## Train + Analyze

```bash
python src/train.py --data-path /Users/andyacchen/Downloads/cleaned_batting_stats.csv
```

This script performs:
1. Dataset intro and supervised target creation (`HR_next_season`)
2. EDA (target distribution, 4+ visualizations, correlation heatmap)
3. Model training/comparison:
   - Linear Regression
   - Decision Tree Regressor (GridSearchCV)
   - Random Forest Regressor (GridSearchCV)
   - XGBoost Regressor (GridSearchCV)
   - MLP Regressor
4. Test metrics (MAE, RMSE, R²)
   - Uses `train_test_split(test_size=0.2, random_state=42)`
5. Model comparison table + bar chart
6. SHAP analysis on the best tree-based model:
   - summary plot
   - bar plot
   - waterfall plot
7. Saved model artifacts
8. 2025 HR predictions from 2024 stats

## Run Streamlit App

```bash
streamlit run app.py
```

Tabs:
- Executive Summary
- Descriptive Analytics
- Model Performance
- Explainability & Interactive Prediction

## Output Artifacts

- `reports/dataset_intro.json`
- `reports/eda_interpretations.txt`
- `reports/model_comparison.csv`
- `reports/best_hyperparameters.json`
- `reports/predicted_2025_hr.csv`
- `reports/figures/*.png`
  - Includes `pred_vs_actual_<model>.png` for each model
- `reports/shap_artifacts/*.png`
- `models/*.joblib`
- `models/metadata.json`
