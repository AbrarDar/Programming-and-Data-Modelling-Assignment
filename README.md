# 🛍️ FastBuy Customer Segmentation — ML Pipeline

> A complete CRISP-DM data analytics pipeline that automates customer segmentation for FastBuy, an e-commerce retailer. Three machine learning models are trained and evaluated to classify customers into **Budget**, **Standard**, and **Premium** tiers.

---

## 📌 Project Overview

FastBuy holds transactional data across two fiscal years in different formats. This project:

- Loads, cleans, and merges both datasets into a single 1,000-record pipeline
- Performs structured Exploratory Data Analysis (EDA) with 6 publication-quality figures
- Trains and evaluates three supervised classifiers: Random Forest, Gradient Boosting, and KNN with SMOTE
- Selects **Gradient Boosting** as the recommended production model (CV accuracy 99.8%, log loss ≈ 0.000)

---

## 📂 Repository Structure

```
FastBuy-Customer-Segmentation/
│
├── Final_Code.ipynb                  # Main Jupyter Notebook (all code + markdown)
│
├── Datasets/
│   ├── FastBuy_2023-2024.csv         # Source data: Sep 2023 – Aug 2024 (488 records)
│   ├── FastBuy_2024-2025.xlsx        # Source data: Sep 2024 – Aug 2025 (512 records)
│   └── combined_dataset.csv          # Auto-generated merged output (audit trail)
│
├── Figures/                          # All 12 EDA & evaluation figures (auto-saved by notebook)
│   ├── Figure 1 - Customer Segment Distribution.png
│   ├── Figure 2 - Annual Income Distribution & by Segment.png
│   ├── ...
│   └── Figure 12 - F1 Score by Segment — All Three Models.png
│
└── README.md
```

---

## 📊 Dataset Description

| File | Format | Period | Records | Notes |
|------|--------|--------|---------|-------|
| `FastBuy_2023-2024.csv` | CSV | Sep 2023 – Aug 2024 | 488 | Labelled `Segment` column |
| `FastBuy_2024-2025.xlsx` | Excel | Sep 2024 – Aug 2025 | 512 | Sales team maintained |

**Features:** `CustomerID`, `Age`, `Gender`, `AnnualIncome`, `SpendingScore`, `PurchaseHistory`, `Location`, `PreferredDevice`, `PaymentMethod`, `PurchaseDate`, `Segment`

**Target variable:** `Segment` — three classes: *Budget* (60.2%), *Standard* (28.4%), *Premium* (11.4%)

---

## ⚙️ Pipeline Stages

The notebook follows the **CRISP-DM** methodology across 18 sections:

| Section | Description |
|---------|-------------|
| 1–2 | Library imports & data loading |
| 3–5 | Data quality assessment, cleaning functions, and transformation |
| 6 | Dataset merging and audit trail export |
| 7 | Exploratory Data Analysis (Figures 1–6) |
| 8 | Feature engineering, label encoding, train/test split |
| 9 | Model 1 — Random Forest Classifier |
| 10 | Model 2 — Gradient Boosting Classifier |
| 11 | Model 3 — KNN with SMOTE Pipeline |
| 12–16 | Evaluation: confusion matrices, feature importance, CV comparison, F1 analysis |
| 17–18 | Final summary table & conclusions |

---

## 🧹 Data Quality Issues Resolved

| Issue | Dataset | Fix Applied |
|-------|---------|-------------|
| Mixed Gender encoding (`'F'`/`'Female'`) | CSV | `clean_gender()` function |
| K-suffix income values (`'41.288 K'`) | CSV | `clean_income()` with ×1000 conversion |
| Duplicate `CustomerID` records | Excel | `.drop_duplicates()` |
| Non-numeric `SpendingScore` entries | Both | Range validation + coercion |

---

## 🤖 Model Results

| Model | Accuracy | Log Loss | CV Accuracy | Recommended Role |
|-------|----------|----------|-------------|------------------|
| **Gradient Boosting** | High | ≈ 0.000 | 99.8% | ✅ Primary — Production |
| Random Forest | High | 0.1627 | ~99% | 🔄 Validation / Backup |
| KNN (SMOTE pipeline) | ~78% | 0.4689 | ~78% | 📊 Baseline / Benchmark |

> **Why Gradient Boosting?** Near-zero log loss means its probability estimates are reliable for routing borderline customers to human review — a key requirement for production deployment.

---

## 🔑 Key Findings

- **SpendingScore** is the single most discriminative feature (r = 0.48 with Segment)
- **AnnualIncome** is the second strongest predictor (r = 0.38), with a clear monotonic gradient across tiers
- **No gender bias** detected — Female vs Male average SpendingScore differs by only 0.4 points
- **5.3× class imbalance** (Budget:Premium) requires `class_weight='balanced'` for RF/GB and SMOTE for KNN
- All results cross-validated against Microsoft Excel (AVERAGEIF / COUNTIF)

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn openpyxl
```

### Running the Notebook

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/fastbuy-customer-segmentation.git
   cd fastbuy-customer-segmentation
   ```

2. Place the dataset files in a `Datasets/` folder and create a `Figures/` folder:
   ```bash
   mkdir Datasets Figures
   # Copy FastBuy_2023-2024.csv and FastBuy_2024-2025.xlsx into Datasets/
   ```

3. Open and run the notebook:
   ```bash
   jupyter notebook Final_Code.ipynb
   ```

4. **Run all cells in order** (Kernel → Restart & Run All). Figures are saved automatically to `Figures/`.

> 💡 The notebook was developed and tested on **Google Colab**. All required libraries are pre-installed there except `imbalanced-learn`, which the notebook installs automatically.

---

## 📚 Technologies Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, merging, and EDA |
| `numpy` | Numerical operations and array handling |
| `matplotlib` / `seaborn` | All visualisations (Figures 1–12) |
| `scikit-learn` | ML models, preprocessing, evaluation metrics |
| `imbalanced-learn` | SMOTE oversampling and `imblearn.Pipeline` |
| `openpyxl` | Reading `.xlsx` files |

---

## 📄 Academic Context

This project was submitted as part of the **MSc Management with Data Analytics** programme at BPP University for the module *Programming & Data Modelling* (Summative Assessment, March 2026).

---

## 📝 License

This repository is for educational and portfolio purposes. Dataset files are proprietary to the assignment and should not be redistributed.
