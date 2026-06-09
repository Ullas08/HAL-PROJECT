# Aircraft Equipment Fault Detection

> **HAL – Helicopter Division** | B.Tech CSE, 6th Sem | Academic Year 2026–27  
> Student: Ullas.T &nbsp;|&nbsp; Guide: Jagadesh.G &nbsp;|&nbsp; Developed for Antigravity

An end-to-end Machine Learning pipeline that ingests aircraft sensor telemetry, trains **six classification algorithms**, selects the best model (target F1 ≥ 0.90), and surfaces predictions through an interactive **Streamlit dashboard**.

---

## 📁 Repository Structure

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
├── models/              # serialised .pkl files
├── app/
│   └── app.py           # Streamlit dashboard
├── tests/               # pytest unit tests
├── reports/             # PNG charts, metrics CSV, final report
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Create & activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline (synthetic dataset — no download needed)

```bash
python src/run_pipeline.py --dataset synthetic
```

This will:
- Generate a 5,000-row synthetic aircraft sensor dataset
- Run preprocessing (imputation → IQR capping → scaling → SMOTE → feature selection)
- Train and tune 6 ML models with GridSearchCV (5-fold CV)
- Evaluate on the test set and generate all charts to `reports/`
- Save the champion model to `models/champion_model.pkl`

### 4. Launch the Streamlit dashboard

```bash
streamlit run app/app.py
```

Open http://localhost:8501 in your browser.

---

## 🌐 Using NASA C-MAPSS Dataset (Primary)

1. Download from: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
2. Extract `CMAPSSData.zip` and copy `train_FD001.txt` to `data/raw/`
3. Run:
   ```bash
   python src/run_pipeline.py --dataset cmapss
   ```

---

## 🤖 ML Algorithms

| Algorithm | Key Hyperparameters Tuned |
|---|---|
| Logistic Regression | C |
| Decision Tree | max_depth, min_samples_split |
| Random Forest | n_estimators, max_depth |
| SVM | C, kernel, gamma |
| K-Nearest Neighbours | n_neighbors, metric |
| XGBoost | n_estimators, learning_rate, max_depth |

All models tuned with `GridSearchCV(scoring='f1', cv=5, n_jobs=-1)`.

---

## ✅ Running Tests

```bash
pytest tests/ -v
```

Expected: **100% pass rate** across 3 test modules (preprocessing, training, app logic).

---

## 🎯 Success KPIs

| Metric | Target |
|---|---|
| Champion F1 (test set) | ≥ 0.90 |
| Champion ROC-AUC | ≥ 0.92 |
| Streamlit load time | < 5 seconds |
| Prediction response time | < 2 seconds |
| pytest pass rate | 100% |
| ML algorithms compared | 6 |

---

## 📊 Deliverables

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

## 📚 References

1. Saxena, A. et al. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* IEEE PHM Conference.
2. Jardine, A. K. et al. (2006). *A review on machinery diagnostics and prognostics.* Mechanical Systems and Signal Processing, 20(7).
3. Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR, 12, 2825–2830.
4. Chawla, N. V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR, 16, 321–357.
5. NASA Prognostics CoE: https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository
