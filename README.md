# Aliexpress-pricing-strategy-ML
ML applied in e-commerce .Pricing strategy analysis of AliExpress KSA sellers using 2 ML pipelines : clustering and classification . 
## Author
Takwa Bennjima 
Yahya Mabrouk 
## Project Overview
This project is an end-to -end Machine Learning analysis of the famous e-commerce site AliExpress KSA .Product listings were scraped from "AliExpress KSA " in 2022.
The problematic quection : 
> **what pricing strategies do sellers adopt on AliExpress KSA ?
> **Which strategy ensures the most sales per product category ?
Instead of classic price prediction, this project analyses seller behaviour through a **2-stage ML pipeline**:
1. **Clustering** - groups sellers into pricing archetypes
2. **Classification** -predicts which archetype dominates each product category


---

## Repository Structure
```
aliexpress-pricing-strategy-ml/
├── data/
│   ├── data.csv             # Raw dataset
│   └── data_eng.csv       # Enriched dataset(engineered data)
├── notebooks/
│   ├── EDA.ipynb              # Exploratory Data Analysis + Feature Engineering
│   └── Modeling.ipynb         # Preprocessing + Clustering + Classification
├── src/                          # (Bonus) Modular Python scripts
├── requirements.txt              # Python dependencies
├── README.md
└── .gitignore
```
---
## Installation & Setup
```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/aliexpress-pricing-strategy-ml.git
cd aliexpress-pricing-strategy-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter
jupyter lab
```
---
## ML Pipeline

### Stage 1 — Clustering (Unsupervised)
- Algorithm: K-Means
- Goal: Discover seller pricing archetypes from features like price, discount, sold, shippingCost
- Selection of k: Elbow Method + Silhouette Score

### Stage 2 — Classification (Supervised)
- Algorithms: Logistic Regression, Random Forest, XGBoost
- Goal: Predict which pricing archetype wins per product category
- Metrics: F1-Score, AUC-ROC, Confusion Matrix
- Optimisation: GridSearchCV on best model
- Interpretation: SHAP values

---



