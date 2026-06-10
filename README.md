# SuperKart Sales Forecasting (Regression + Deployment)

> Forecast per-product store revenue with Gradient Boosting & XGBoost, then serve it via a Flask API + Streamlit UI on Hugging Face.


## Problem

SuperKart, a retail chain, wants to forecast the next quarter's sales revenue for each product in each outlet to optimize inventory and regional strategy.


## At a glance

- **Problem type:** Supervised Regression + Model Deployment
- **Target:** `Product_Store_Sales_Total` (revenue, continuous)
- **Primary metric(s):** RMSE (primary), MAE, R², Adjusted R², MAPE
- **Tech stack:** Python, pandas, NumPy, scikit-learn (Pipeline, ColumnTransformer, GridSearchCV), XGBoost, Flask, Streamlit, Docker, Hugging Face Spaces, joblib


## Data

`SuperKart.csv` (~0.86 MB) — product attributes (weight, sugar content, allocated area, type, MRP) and store attributes (establishment year, size, city tier, store type).


## Approach / Steps

1. Data overview & type checks
2. EDA — univariate & bivariate analysis of products and stores
3. Data preprocessing — feature engineering, `make_column_transformer` (OneHotEncoder + StandardScaler)
4. Model building — Decision Tree, Bagging, Random Forest, AdaBoost, Gradient Boosting, XGBoost regressors
5. Hyperparameter tuning with `GridSearchCV`
6. Model comparison on RMSE / MAE / R² / Adjusted R² / MAPE; serialize best model with `joblib`
7. Backend deployment — Flask REST API (`app.py`) + Dockerfile
8. Frontend deployment — Streamlit UI (online + batch CSV prediction) + Dockerfile
9. Push backend & frontend to Hugging Face Spaces (Docker) via `huggingface_hub`


## Repo structure

```
superkart-sales-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
├── data/
├── backend/
├── frontend/
```


## How to run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook   # open the notebook in notebooks/
```

## Sharing the links of my Hugging Face Spaces
Frontend: https://huggingface.co/spaces/runtimeRD/Superkart_Salesrevenue_Frontend

Backend: https://huggingface.co/spaces/runtimeRD/SuperKartSalesRevenuePredictionBackend


## License

Private project — Not licensed for distribution.
