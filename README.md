
# cmse802_project
## Overview
This project builds a full, reproducible machine-learning pipeline to predict used-car prices from the Craigslist vehicles dataset.
It showcases:

- Automated data cleaning
- Unified feature engineering
- Modern EDA with Plotly
- A modular src/ architecture
- GridSearch-tuned Gradient Boosting Regressor (GBR)
- Reproducible command-line workflow
- Organized metrics, plots, and artifacts in `results/`

This is the final, simplified, optimized version of the project (GBR-only).

## Final Workflow Summary
The entire pipeline is executed with:

python -m src.run_experiment

The script performs:

1. Load raw CSV
2. Clean dataset
3. Feature-engineering / preprocessing
4. Automated EDA
5. Train/test split
6. GridSearchCV hyperparameter tuning
7. Train GradientBoostingRegressor
8. Export:
   - Feature importances
   - Parity & residual plots
   - Train/test metrics
   - Cross-validation logs
   - Best model (joblib)
9. Streamlit app developed (Link At the end)
Older features (multi-model training, randomized search, JSON logs, Streamlit deployment) were intentionally removed.

## Folder Structure (Final)
cmse802_project/
    src/
        data_clean.py
        features.py
        eda.py
        preprocess.py
        registry.py
        plots.py
        run_experiment.py
        __init__.py
    results/
        gbr/
            best_model.joblib
            feature_importances.csv
            train_test_metrics.csv
            parity_train.png
            parity_test.png
            residuals_hist_test.png
            residuals_vs_pred_test.png
        all_cv_runs.csv
        eda/
            distributions_*.html
            correlation_heatmap.html
            trends_*.html
    notebooks/demo.ipynb
    data/
    tests/
    app.py
    requirements.txt
    README.md

## Features Removed (Earlier Versions)
- RandomForest, SVR, Linear, MLP models
- RandomizedSearch → replaced with GridSearchCV
- Leaderboards & multi-model comparison
- JSON metric files
- Log-price model variants
- Old scripts (`train_ml.py`, `train_regress.py`, `evaluate.py`)
- Streamlit deployment as a requirement

## How to Run
1. pip install -r requirements.txt
2. python -m src.run_experiment

Optional flags:
--use_pca
--pca_components N
--use_log_price
--test_size 0.2
--random_state 42
--results_dir results

## Outputs
results/
    all_cv_runs.csv
    eda/*.html
    gbr/
        best_model.joblib
        feature_importances.csv
        train_test_metrics.csv
        parity_train.png
        parity_test.png
        residuals_hist_test.png
        residuals_vs_pred_test.png

## Why Gradient Boosting Only?
GBR consistently gave:
- Highest R²
- Lowest MAE & RMSE
- Most stable generalization
- Clean feature importances
- Strong performance on full dataset

## Summary
A clean, production-style ML pipeline with:
- Automated EDA
- Structured preprocessing
- Tuned GBR model
- Reproducible CLI execution
- Organized metrics, plots, artifacts
- Professional file hierarchy
- Streamlit app connected to use the model in a user friendly way.
### CHATGPT 5.1 was used in various parts of the project to make it easier to modify and understand

To use Streamlit app
## https://cmse802project-n3ecnyfnkczizedjswnvcj.streamlit.app/
