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

## Week 1: Setup, Role Alignment & Data Familiarization  
**Dates:** Nov 17 – Nov 23  

### Goals
- Finalize roles and workflow setup  
- Load, explore, and clean the `public_cases.json` dataset  
- Understand project requirements from PRD and interview documents  

### Task Breakdown

| Role | Tasks |
|------|-------|
| **Saniyah (Business Analyst)** | - Review `PRD.md` and `INTERVIEWS.md` from the challenge repo<br>- Summarize key business rules, phrases, and hints about reimbursement logic<br>- Begin drafting `reports/business_logic_hypothesis.qmd` with early ideas about how reimbursement might be calculated |
| **Vy (Data Scientist)** | - Load dataset into `01_eda.qmd`<br>- Create data summaries using `data.describe()`<br>- Generate distribution plots (trip duration, miles, receipts, reimbursement)<br>- Identify potential outliers and missing values |
| **Harriet (ML Engineer)** | - Review data outputs from Vy<br>- Research suitable model candidates (Linear Regression, Decision Tree, Random Forest, Gradient Boosting)<br>- Draft plan for model comparison criteria (MAE, RMSE, $±0.01$ and $±1.00$ thresholds) |
| **Rehinatu (Software Engineer)** | - Set up project structure and confirm repo organization (data, models, notebooks, reports)<br>- Verify that everyone can render `.qmd` files in Quarto<br>- Create initial `predict.py` file with placeholder function that accepts 3 inputs and returns a dummy reimbursement |

### Deliverables
- `01_eda.qmd` initial version with visualizations  
- `business_logic_hypothesis.qmd` first draft  
- Repo and branch setup verified  
- Functional `predict.py` placeholder  

---

## Week 2: Feature Engineering & Baseline Model Implementation  
**Dates:** Nov 24 – Nov 30  

### Goals
- Engineer new features for better pattern detection  
- Build and test baseline models  
- Begin documenting relationships between variables and outcomes  

### Task Breakdown

| Role | Tasks |
|------|-------|
| **Saniyah** | - Add interpretive paragraphs for EDA findings in report<br>- Explain trends (e.g., how miles vs receipts may impact reimbursement)<br>- Connect early findings to possible business policies (e.g., per-mile rate, minimum daily cap) |
| **Vy** | - Create new features (`cost_per_mile`, `cost_per_day`, `receipts_ratio`)<br>- Normalize or scale data if needed<br>- Document feature engineering steps in `02_feature_engineering.qmd` |
| **Harriet** | - Implement **Linear Regression** and **Decision Tree Regressor** in `03_models.qmd`<br>- Evaluate both using Mean Absolute Error and Root Mean Square Error<br>- Save outputs and metrics tables for report inclusion |
| **Rehinatu** | - Update `predict.py` to load the model file (once trained)<br>- Add argument parsing to take 3 command-line inputs<br>- Begin writing README instructions for running predictions |

### Deliverables
- `02_feature_engineering.qmd` completed  
- `03_models.qmd` baseline model results with comparison  
- Updated `predict.py` with CLI interface  
- Updated report with EDA interpretations and early business insights  

---

## Week 3: Advanced Models, Model Tuning & Interpretability  
**Dates:** Dec 1 – Dec 4  

### Goals
- Improve model performance through tuning and feature selection  
- Introduce ensemble or non-linear methods  
- Interpret model behavior and extract insights  

### Task Breakdown

| Role | Tasks |
|------|-------|
| **Saniyah** | - Translate technical results into business terms for the report<br>- Draft interpretability section (e.g., "Mileage explains 70% of variation")<br>- Write summary of key findings from EDA + model comparison |
| **Vy** | - Conduct correlation and feature importance analysis<br>- Visualize top predictive features (bar plots, heatmaps)<br>- Experiment with polynomial or interaction features |
| **Harriet** | - Implement **Random Forest**, **XGBoost**, and **Ridge Regression** models<br>- Conduct hyperparameter tuning (using GridSearchCV)<br>- Compare performance and interpret feature importance via SHAP or permutation importance |
| **Rehinatu** | - Integrate best-performing model into `predict.py`<br>- Test prediction timing (must run < 5 seconds)<br>- Add error handling (invalid input types, negative values, etc.) |

### Deliverables
- `03_models.qmd` with 3–4 models compared  
- Interpretation visuals (SHAP plots, feature rankings)  
- Updated `predict.py` with optimized model loading and input validation  
- Draft of interpretability write-up in `reports/final_report.qmd`  

---

## Week 4: Integration, Final Report & Presentation Prep  
**Dates:** Dec 5 – Dec 8  

### Goals
- Combine all results into a cohesive report  
- Prepare the final presentation and demonstration  
- Ensure all code is reproducible and documented  

### Task Breakdown

| Role | Tasks |
|------|-------|
| **Saniyah** | - Finalize written sections (Introduction, Business Insights, Conclusion)<br>- Create presentation slides (overview, results, and insights)<br>- Record or narrate business explanation segment for presentation video |
| **Vy** | - Double-check dataset transformations and visualizations for clarity<br>- Add clean charts and tables to `final_report.qmd`<br>- Proofread technical sections for accuracy |
| **Harriet** | - Ensure final model is saved as `models/final_model.pkl`<br>- Document model selection logic and hyperparameters<br>- Contribute to technical section of presentation (explain models and evaluation metrics) |
| **Rehinatu** | - Test final prediction pipeline with example inputs<br>- Finalize `README.md` setup instructions and usage examples<br>- Push all final versions of `.qmd`, `.html`, and `.py` files to `main` branch |

### Deliverables
- `reports/final_report.qmd` + rendered `final_report.html`  
- `predict.py` finalized and tested  
- Presentation video (20 minutes) uploaded or shared  
- Repository finalized and submitted by **Dec 8**  

---

## Weekly Meeting Schedule
| Week | Meeting Focus | Primary Owner |
|------|----------------|----------------|
| Week 1 | Setup, data understanding, assign roles | All |
| Week 2 | EDA + baseline results review | All |
| Week 3 | Model comparison & tuning discussion | All |
| Week 4 | Final integration & presentation rehearsal | All |

---

---

## Progress Tracking
- Each member should **push commits to their branch** by Sunday each week.  
- Create **pull requests to `main`** by Sunday evening after review.  
- Share what they have completed in the Teams chat and next steps they are planning to do.   

---




