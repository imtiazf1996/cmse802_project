# cmse802_project
## Overview
This project predicts used car prices from features such as make, model, year, mileage, and condition.  
The goal is to apply computational modeling methods from CMSE 802, including data preprocessing, numerical modeling, supervised machine learning, unit testing, and reproducible command-line workflows.  

## Objectives
- Clean and preprocess a car dataset (handle missing values, encode categorical features).  
- Implement a baseline linear regression model for price prediction.  
- Train advanced supervised ML models (Random Forest, Gradient Boosting) and compare performance.  
- Create interactive visualizations (Plotly) to show price depreciation vs. year and mileage.  
- Ensure reproducibility with GitHub, CLI workflows, and testing.
## Folder Structure
```

cmse802_project/
│
├── src/               # Core Python scripts
│   ├── data_clean.py
│   ├── eda.py
│   ├── train_regress.py
│   ├── train_ml.py
│   └── evaluate.py
│
├── notebooks/         # Jupyter notebooks for analysis & visualization
│   └── demo.ipynb
│
├── data/              # Input data (large CSVs kept local/ignored)
│
├── tests/             # Unit tests (structure present)
│
├── results/           # Generated outputs (plots, metrics, model files)
│
├── docs/              # Documentation
│
├── app.py             # Streamlit EDA app
├── requirements.txt   # Dependencies
└── README.md

```
```
    A[Raw Data (vehicles.csv)] --> B[data_clean.py]
    B --> C[Cleaned Data]
    C --> D[train_regress.py]
    C --> E[train_ml.py]
    D --> F[Baseline Model Outputs]
    E --> F
    F --> G[evaluate.py]
    G --> H[results/ (metrics, plots, models)]
    H --> I[notebooks/demo.ipynb]
    H --> J[app.py (Streamlit EDA)]
Scripts can be run individually

python src/data_clean.py<br>
python src/eda.py<br>
python src/train_regress.py<br>
python src/train_ml.py<br>
python src/evaluate.py
