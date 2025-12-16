# CSCI / DASC 6020: Machine Learning Team Project
### *Black Box Legacy Reimbursement System (Top Coder Challenge)*

**Authors:** Harriet O’Brien, Vy Tran, Saniyah Khan, Rehinatu Usman  
**Course:** CSCI / DASC 6020 — Machine Learning  
**Date:** Fall 2025  

---

## Project Overview
This project is based on the [Top Coder Challenge: Black Box Legacy Reimbursement System](https://github.com/8090-inc/top-coder-challenge).  
The objective is to **reverse-engineer a 60-year-old travel reimbursement system** for **ACME Corporation** using only historical data and employee interviews.

Teams act as ML consultants tasked with discovering the **hidden business logic** of the legacy system and developing a **predictive model** that replicates its reimbursement behavior.

**Scenario: Your team has been hired as ML consultants by ACME Corporation. Their legacy reimbursement system has been running for 60 years, no one knows how it works, but it is still used daily. A new system has been built, but the ACME Corporation is confused by the differences in results. Your mission is to use machine learning to understand the original business logic and create a model that can explain and predict the legacy system’s behavior.**


---

## Project Goals
By completing this project, we aim to:

- Apply **supervised learning** methods to model complex business rules  
- Perform **exploratory data analysis (EDA)** and **feature engineering** to uncover trends  
- Build **interpretable ML models** that explain legacy decision logic  
- Communicate results clearly to both **technical** and **non-technical** stakeholders  
- Collaborate effectively using **GitHub** and **Quarto notebooks**

---

## Dataset
- **File:** `data/public_cases.json`  
- **Size:** 1,000 historical reimbursement cases  
- **Inputs:**
  - `trip_duration_days` — Days spent traveling  
  - `miles_traveled` — Distance traveled  
  - `total_receipts_amount` — Total receipts in USD  
- **Output:**
  - `reimbursement` — Legacy system reimbursement amount (float, rounded to 2 decimals)

---

## Technical Phases

### **Phase 1 — Exploratory Data Analysis (Weeks 1–2)**
- Statistical summary and visualizations  
- Correlation and outlier detection  
- Business logic hypothesis development  

### **Phase 2 — Model Development (Weeks 3–5)**
- Implementation of 4+ ML models (Linear, Tree-based, Ensemble, etc.)  
- Model evaluation and interpretability  

### **Phase 3 — System Integration (Weeks 6–7)**
- Develop and finalize the `predict.py` script.
- Load the serialized model (`final_model.pkl`) and return a single prediction.
- Add input validation, error handling, and runtime checks.
- Ensure the script runs in under 5 seconds per test case.
- Test model outputs with various input combinations.
- Finalize GitHub documentation and ensure all Quarto notebooks render correctly. 

### **Phase 4 — Business Communication (Week 8)**
- Technical report (15–20 pages)  
- Business presentation (20 min recorded)  
- Final GitHub repository submission with Quarto notebooks  

---

## Team Roles

| Role | Responsibilities |
|------|------------------|
| **Data Scientist / Analyst** | Leads EDA, feature engineering, and visualization |
| **ML Engineer** | Builds and tunes ML models |
| **Business Analyst** | Analyzes PRD & interviews, interprets results, writes business insights |
| **Software Engineer** | Develops production-ready code for model integration. Implements 'prediction.py' for inference, manages preprocessing and post-processing pipelines, ensures code quality and reproducibility |

---

# 6020 ML Project: Team Schedule & Task Breakdown
**Project:** Black Box Legacy Reimbursement System  
**Deadline:** December 8, 2025  
**Team Members:**  
- **Harriet O’Brien** — Machine Learning Engineer  
- **Vy Tran** — Data Scientist  
- **Saniyah Khan** — Business Analyst  
- **Rehinatu Usman** — Software Engineer  

---

## Progress Tracking
- Each member should **push commits to their branch** by Sunday each week.  
- Create **pull requests to `main`** by Sunday evening after review.  
- Share what they have completed in the Teams chat and next steps they are planning to do.   

---

# How to Run This Project 

### **Environment:**
- Python 3.9+
- Required packages: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `shap`, `joblib`, `pyyaml`

### **Setup:**
- 
### **Run Modeling:**
- 
### **Run XGBoost:** 
- 
### **Run SHAP analysis:**

## Data Science Workflow
The data science portion of this project is implemented using Jupyter notebooks.
Here are the pre-rendered HTML document for ease of viewing, please click to view. 

[View Exploratory Data Analysis](https://khan2225.github.io/6020_MLProject/notebooks/01_eda.html)

[View Feature Engineering](https://khan2225.github.io/6020_MLProject/notebooks/02_feature_engineering.html)

[View Models](https://khan2225.github.io/6020_MLProject/notebooks/03_models.html)

[View Interpretability](https://khan2225.github.io/6020_MLProject/notebooks/04_interpretabillity.html)

To also reproduce the modeling and analysis results, run the notebooks in the following order:

1. 01_eda.ipynb– Exploratory data analysis and initial data inspection  
2. 02_feature_engineering.ipynb – Feature creation and preprocessing  
3. 03_models.ipynb – Model training and performance comparison  
4. 04_interpretability.ipynb – Model interpretation and validation
   
Each notebook can be run sequentially without additional configuration.
The final trained model is exported as final_model.pkl for downstream use.

## Business Logic Hypothesis 
The Business Logic Hypothesis is provided as a pre-rendered HTML document for ease of viewing. 

[View Business Logic](https://khan2225.github.io/6020_MLProject/reports/business_logic_hypothesis.html)

The corresponding `.qmd` file is included in the repository for more viewing as well. 

