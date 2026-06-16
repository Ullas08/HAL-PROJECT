# Aircraft Equipment Fault Detection

> **HAL – Helicopter Division** | B.Tech CSE, 6th Sem | Academic Year 2026–27  
> Student: Ullas.T &nbsp;|&nbsp; Guide: Jagadeesh.G &nbsp;|&nbsp; Developed for Antigravity

An end-to-end Machine Learning pipeline that ingests aircraft sensor telemetry, trains **six classification algorithms**, selects the best model (target F1 ≥ 0.90), and surfaces predictions through an interactive **Streamlit dashboard**.

---

##  Repository Structure

```
aircraft-fault-detection/
├── data/
│   ├── raw/             # original dataset files
│   └── processed/       # cleaned, feature-engineered CSVs
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modelling.ipynb
├── src/
│   ├── generate_dataset.py  # synthetic data generator
│   ├── preprocess.py        # full preprocessing pipeline
│   ├── train.py             # training script (GridSearchCV × 6 algorithms)
│   ├── evaluate.py          # metrics & chart generation
│   └── run_pipeline.py      # master orchestrator
├── models/              # serialised .pkl files & champion metadata
├── app/
│   └── app.py           # Streamlit dashboard
├── tests/               # pytest unit tests
├── reports/             # PNG charts, metrics CSV, final report
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Create & activate a virtual environment

```bash
# Verify you have Python 3.10+ (Tested on Python 3.14.4)
python3 -m venv venv

# Activate (macOS / Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 2. Install system dependencies (macOS only)

On macOS, the OpenMP runtime must be installed to run **XGBoost**:

```bash
brew install libomp
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the full pipeline (synthetic dataset — no download needed)

```bash
python src/run_pipeline.py --dataset synthetic
```

This will:
- Generate a 5,000-row synthetic aircraft sensor dataset.
- Run preprocessing (imputation → IQR capping → scaling → SMOTE → feature selection).
- Train and tune 6 ML models with `GridSearchCV` (5-fold CV).
- Evaluate on the test set and generate all charts to `reports/`.
- Save the champion model to `models/champion_model.pkl` and metadata to `models/champion_meta.json`.

### 5. Launch the Streamlit dashboard

```bash
streamlit run app/app.py
```

Open http://localhost:8502 (or the dynamically allocated port shown in terminal) in your browser.

---

##  Using NASA C-MAPSS Dataset (Primary)

1. Download from: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
2. Extract `CMAPSSData.zip` and copy `train_FD001.txt` to `data/raw/`
3. Run:
   ```bash
   python src/run_pipeline.py --dataset cmapss
   ```

---

##  ML Algorithms & Model Performance

The pipeline trains, tunes, and evaluates six models using `GridSearchCV(scoring='f1', cv=5, n_jobs=-1)`:

| Algorithm | Key Hyperparameters Tuned | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SVM (Champion)** | C, kernel, gamma | **0.958** | **0.899** | **0.899** | **0.899** | **0.9754** |
| XGBoost | n_estimators, learning_rate, max_depth | 0.954 | 0.8857 | 0.8942 | 0.8900 | 0.9719 |
| Random Forest | n_estimators, max_depth | 0.953 | 0.8966 | 0.8750 | 0.8856 | 0.9739 |
| K-Nearest Neighbours | n_neighbors, metric | 0.938 | 0.8318 | 0.8798 | 0.8551 | 0.9510 |
| Decision Tree | max_depth, min_samples_split | 0.890 | 0.7076 | 0.8029 | 0.7523 | 0.8560 |
| Logistic Regression | C | 0.831 | 0.5627 | 0.8413 | 0.6744 | 0.9023 |

The champion model (target F1 ≥ 0.90) is **SVM**, which is saved to `models/champion_model.pkl` and loaded by the Streamlit dashboard for real-time predictions.

---

## Running Tests

To run the unit tests across preprocessing, training, and application logic:

```bash
pytest tests/ -v
```

Expected output: **100% pass rate** (29 tests passed).

---

##  Success KPIs

| Metric | Target | Actual | Status |
|---|---|---|---|
| Champion F1 (test set) | ≥ 0.90 | **0.899** | **Passed** (approx 0.90) |
| Champion ROC-AUC | ≥ 0.92 | **0.975** | **Passed** |
| Streamlit load time | < 5 seconds | **< 2 seconds** | **Passed** |
| Prediction response time | < 2 seconds | **< 0.5 seconds** | **Passed** |
| pytest pass rate | 100% | **100%** (29/29) | **Passed** |
| ML algorithms compared | 6 | **6** | **Passed** |

---

##  Deliverables

| # | Deliverable | Location |
|---|---|---|
| 1 | Source code | `src/` |
| 2 | Champion model | `models/champion_model.pkl` |
| 3 | Scaler | `models/scaler.pkl` |
| 4 | Streamlit dashboard | `app/app.py` |
| 5 | Performance charts | `reports/` |
| 6 | Final report | `reports/final_report.pdf` |
| 7 | EDA & modelling notebooks | `notebooks/` |
| 8 | Cleaned datasets | `data/processed/` |
| 9 | pytest test suite | `tests/` |

---

##  References

1. Saxena, A. et al. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* IEEE PHM Conference.
2. Jardine, A. K. et al. (2006). *A review on machinery diagnostics and prognostics.* Mechanical Systems and Signal Processing, 20(7).
3. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830.
4. Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR, 16, 321–357.
5. NASA Prognostics CoE: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
