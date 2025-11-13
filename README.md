# CSCI / DASC 6020: Machine Learning Team Project
### *Black Box Legacy Reimbursement System (Top Coder Challenge)*

**Authors:** Harriet O’Brien, Vy Tran, Saniyah Khan  
**Course:** CSCI / DASC 6020 — Machine Learning  
**Date:** Fall 2025  

---

## Project Overview
This project is based on the [Top Coder Challenge: Black Box Legacy Reimbursement System](https://github.com/8090-inc/top-coder-challenge).  
The objective is to **reverse-engineer a 60-year-old travel reimbursement system** for **ACME Corporation** using only historical data and employee interviews.

Teams act as ML consultants tasked with discovering the **hidden business logic** of the legacy system and developing a **predictive model** that replicates its reimbursement behavior.

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
- Create production-ready prediction script `predict.py`  
- Develop preprocessing and post-processing pipelines  

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


---




