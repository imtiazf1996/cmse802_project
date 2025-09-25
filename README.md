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
├── data/
│ ├── raw/ # raw CSVs (ignored in git)
├── src/ # source scripts
│ ├── data_clean.py
│ ├── eda.py
│ ├── train_regress.py
│ ├── train_ml.py
│ └── evaluate.py
├── tests/ # unit tests (planned)
├── results/ # plots, models, metrics (planned)
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md
```
Scripts can be run individually

python src/data_clean.py
python src/eda.py
python src/train_regress.py
python src/train_ml.py
python src/evaluate.py
