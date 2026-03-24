# 🛍️ FastBuy Customer Segmentation - ML Pipeline

> A complete CRISP-DM data analytics pipeline that automates customer segmentation for FastBuy, an e-commerce retailer. Three machine learning models are trained and evaluated to classify customers into **Budget**, **Standard**, and **Premium** tiers.

---

## 📌 Project Overview

FastBuy is an e-commerce company that manually segmented its customers into three commercial tiers - Budget, Standard, and Premium - to personalise marketing campaigns. This manual process was time-consuming and inconsistent. The goal of this project was to **automate that segmentation using machine learning**, so that any new customer record can be instantly and reliably classified.

The project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology - the industry-standard framework for data science projects - moving through business understanding, data preparation, modelling, and evaluation in a structured, reproducible pipeline.

The full pipeline covers:
- Loading and merging two years of customer data from different file formats (CSV + Excel)
- Identifying and fixing real data quality problems before any analysis
- Exploring the data visually to understand what drives customer segmentation
- Training three different machine learning models and comparing them rigorously
- Selecting **Gradient Boosting** as the recommended production model with 99.8% cross-validated accuracy and near-zero log loss

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
│   └── Figure 12 - F1 Score by Segment - All Three Models.png
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Description

Two datasets were provided by FastBuy's Database Administrator, stored in different formats due to separate team ownership - a common real-world data governance challenge:

| File | Format | Period | Records | Notes |
|------|--------|--------|---------|-------|
| `FastBuy_2023-2024.csv` | CSV | Sep 2023 – Aug 2024 | 488 | Extracted from segmentation team's records system |
| `FastBuy_2024-2025.xlsx` | Excel | Sep 2024 – Aug 2025 | 512 | Maintained manually by the Sales team |

**Features used:** `Age`, `Gender`, `AnnualIncome`, `SpendingScore`, `PurchaseHistory`, `Location`, `PreferredDevice`, `PaymentMethod`, `PurchaseDate`

**Target variable:** `Segment` - three classes:

| Segment | Count | Share |
|---------|-------|-------|
| Budget | 602 | 60.2% |
| Standard | 284 | 28.4% |
| Premium | 114 | 11.4% |

> ⚠️ This 5.3× imbalance between Budget and Premium is a critical finding - a model that simply predicted "Budget" for every customer would still achieve 60% accuracy, which is why raw accuracy alone cannot be trusted as an evaluation metric.

---

## 🧹 What We Did - Step by Step

### Step 1: Data Loading
Both files were loaded into pandas DataFrames and immediately inspected for shape, column names, and data types. This caught structural issues early before they could silently corrupt later analysis.

### Step 2: Data Quality Assessment
Before any transformation, a systematic quality report was run on both datasets. Four real problems were identified and documented:

| Problem Found | Dataset | Why It Matters |
|---------------|---------|----------------|
| Mixed Gender encoding - `'F'` and `'Female'` used interchangeably | CSV | Would create 4 gender groups instead of 2, breaking EDA and model encoding |
| Income stored as strings with a `'K'` suffix (e.g. `'41.288 K'`) | CSV | Pandas reads these as text, making arithmetic operations impossible |
| Duplicate `CustomerID` records | Excel | Inflates customer counts and introduces bias into model training |
| `SpendingScore` values outside the valid 1–100 range | Both | Invalid data passed to a model produces silently wrong predictions |

### Step 3: Cleaning Functions
Two reusable Python functions were written to fix the quality issues:

- **`clean_gender(val)`** - standardises all gender values to `'Female'`, `'Male'`, or `'Other'`, handling any abbreviation or whitespace variation. Designed to be *idempotent* (safe to run multiple times without changing the result).
- **`clean_income(val)`** - detects the `'K'` suffix and multiplies by 1,000 to return a plain float. Non-parseable entries are coerced to `NaN` for safe downstream handling.

Both functions were applied using pandas `.apply()` - a vectorised operation significantly faster than a manual for-loop on large datasets.

### Step 4: Merging the Datasets
A `Source` tag (`'2023-2024'` or `'2024-2025'`) was added to each dataset before combining them vertically with `pd.concat()`. This preserves year-of-origin for any future time-based analysis such as detecting year-on-year segment drift. The merged 1,000-record dataset was exported to `combined_dataset.csv` as a reproducible audit trail.

### Step 5: Exploratory Data Analysis (EDA)
Six figures were produced to understand the data before any modelling began. Key statistics were independently verified in Microsoft Excel using `AVERAGEIF` and `COUNTIF` to confirm the Python pipeline produced correct results - a rigorous double-check rarely done but important for data integrity.

### Step 6: Feature Engineering
All categorical columns (`Gender`, `PurchaseHistory`, `Location`, `PreferredDevice`, `PaymentMethod`) were encoded as integers using `LabelEncoder`. `LabelEncoder` was chosen over One-Hot Encoding because tree-based models (Random Forest, Gradient Boosting) split on thresholds and do not assume any ordering in the encoded integers, making One-Hot Encoding unnecessary and computationally wasteful here.

### Step 7: Train/Test Split
A **stratified 80/20 split** was applied with `random_state=42` for full reproducibility. Stratification ensures the class proportions (60.2% Budget, 28.4% Standard, 11.4% Premium) are preserved in both the training and test sets - essential with a 5.3× class imbalance.

### Step 8–10: Three Models Trained

**Model 1 - Random Forest:** 100 independent decision trees, each trained on a random bootstrap sample of the data. Final predictions are made by majority vote across all trees. `class_weight='balanced'` automatically adjusts each tree to compensate for the Budget/Premium imbalance. No feature scaling required - trees split on thresholds and are invariant to feature scale.

**Model 2 - Gradient Boosting:** 100 trees built *sequentially*, where each new tree focuses on correcting the mistakes of the previous ensemble. This sequential error-correction naturally gives extra attention to hard-to-classify minority records (Premium customers), making it robust to imbalance without explicit class weighting.

**Model 3 - KNN with SMOTE Pipeline:** K-Nearest Neighbours classifies each record by majority vote among its K closest neighbours. It requires both feature scaling (`StandardScaler`) and imbalance handling (`SMOTE` - Synthetic Minority Over-sampling Technique). An `imblearn.Pipeline` was used to ensure SMOTE is only applied to the training portion of each cross-validation fold, preventing data leakage into the validation fold. The optimal K was found by sweeping K=1 to K=20 and selecting the value with the highest 5-fold CV accuracy.

### Step 11–15: Rigorous Evaluation
Each model was evaluated across four complementary metrics:

- **Accuracy** - percentage of correctly classified customers
- **Log Loss** - quality of the model's probability estimates (lower = more trustworthy confidence scores)
- **Macro F1 Score** - per-class precision/recall balance, giving equal weight to all three segments regardless of size
- **5-Fold Stratified Cross-Validation** - performance measured across 5 different data splits to confirm stability and rule out lucky test-set results

---

## 🔍 Key Findings Explained

### Finding 1: SpendingScore is the dominant predictor
The correlation heatmap (Figure 5) shows SpendingScore has the strongest linear relationship with Segment (r = 0.48). Figure 3 makes this even clearer - Budget customers cluster tightly around a low spending score (median ≈ 26), Standard customers in the mid-range (≈ 62), and Premium customers in the upper range (≈ 85), with minimal overlap between groups. This near-perfect separation is the primary reason tree-based models achieve such high accuracy.

### Finding 2: AnnualIncome follows a clear monotonic gradient across tiers
Budget customers earn on average £72,502, Standard customers £101,925, and Premium customers £127,528 (Figure 2). This consistent income staircase across tiers confirms that income is a powerful second predictor. Feature importance scores from both Random Forest and Gradient Boosting (Figure 9) confirm that SpendingScore and AnnualIncome together account for the vast majority of each model's predictive power - all other features contribute relatively little.

### Finding 3: Demographics have almost no predictive power
Age is broadly uniform across all customers (range 18–69, mean 43.8 years). Payment method and preferred device are nearly evenly distributed across all three segments. The correlation heatmap confirms all demographic features have |r| < 0.10 with Segment - effectively zero. This means the model segments customers entirely on financial behaviour, not on who they are. This is actually a positive result - it means the segmentation is commercially meaningful and not a proxy for demographic characteristics.

### Finding 4: No gender bias detected
A fairness check compared the average SpendingScore between Female and Male customers - the difference was only 0.4 points (Female: 49.0, Male: 48.6). The stacked bar chart in Figure 6 shows an almost identical segment distribution across genders (~60% Budget, ~28% Standard, ~12% Premium for both). The trained models are therefore unlikely to produce gender-discriminatory predictions, which is an important compliance consideration under UK GDPR Article 22.

### Finding 5: Gradient Boosting outperforms all models on the metric that matters most in production
While Random Forest and Gradient Boosting achieved near-identical hard-prediction accuracy (~99%), **Gradient Boosting's log loss was near zero** compared to Random Forest's 0.1627 and KNN's 0.7075. In the proposed production system, model-predicted probabilities are used to route borderline cases (low-confidence predictions) to human review. This routing logic depends entirely on well-calibrated probabilities - making log loss the decisive production metric, and Gradient Boosting the clear winner.

### Finding 6: KNN is a useful baseline but not suitable for production
KNN achieved approximately 78% accuracy compared to ~99% for the tree-based models. It also cannot produce feature importance scores, because KNN stores training data rather than learning a parameterised decision function - making it impossible to explain predictions to compliance or audit teams as required under GDPR. KNN is retained in the notebook as a conceptual baseline to demonstrate the value gained by adopting more sophisticated approaches.

---

## 📈 Selected Figures

**Figure 1 - Customer Segment Distribution**
![Figure 1](Figures/Figure%201%20-%20Customer%20Segment%20Distribution.png)

**Figure 2 - Annual Income Distribution & by Segment**
![Figure 2](Figures/Figure%202%20-%20Annual%20Income%20Distribution%20%26%20by%20Segment.png)

**Figure 3 - SpendingScore Distribution & by Segment**
![Figure 3](Figures/Figure%203%20-%20SpendingScore%20Distribution%20%26%20by%20Segment.png)

**Figure 5 - Correlation Heatmap**
![Figure 5](Figures/Figure%205%20-%20Correlation%20Heatmap.png)

**Figure 8 - Confusion Matrices (All Three Models)**
![Figure 8](Figures/Figure%208%20-%20Confusion%20Matrices%20%E2%80%94%20Random%20Forest%20vs%20Gradient%20Boosting%20vs%20KNN.png)

**Figure 9 - Feature Importance (RF & GB)**
![Figure 9](Figures/Figure%209%20-%20Feature%20Importance%20(RF%20%26%20GB%20only%20%E2%80%94%20KNN%20does%20not%20support%20this).png)

**Figure 12 - F1 Score by Segment**
![Figure 12](Figures/Figure%2012%20-%20F1%20Score%20by%20Segment%20%E2%80%94%20All%20Three%20Models.png)

---

## 🤖 Final Model Comparison

| Model | Accuracy | Log Loss | CV Accuracy | Recommended Role |
|-------|----------|----------|-------------|------------------|
| **Gradient Boosting** | ~99% | ≈ 0.000 | 99.8% | ✅ Primary - Production |
| Random Forest | ~99% | 0.1627 | ~99% | 🔄 Validation / Backup |
| KNN (SMOTE pipeline) | ~78% | 0.4689 | ~78% | 📊 Baseline / Benchmark |

> **Why Gradient Boosting wins:** Both RF and GB get nearly identical hard predictions correct. The difference is in *how confident* they are. GB's near-zero log loss means when it says a customer is 92% likely to be Premium, that estimate can be trusted. RF's higher log loss means its confidence scores are less reliable - fine as a backup model, but not ideal as the sole production decision-maker.

---

## 🚀 Getting Started

### Prerequisites

Install all dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install individually:

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
