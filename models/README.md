# Models

Trained model files (`.pkl`) are saved here after running the notebook.

## Expected Files

- `model_svm.pkl` — Support Vector Machine (RBF kernel)
- `model_nb.pkl` — Gaussian Naive Bayes
- `model_xgb.pkl` — XGBoost (GridSearchCV tuned)
- `model_logreg.pkl` — Logistic Regression
- `scaler.pkl` — MinMaxScaler for feature normalization

> **Note:** Model files are excluded from Git via `.gitignore` due to their size. Train models by running the notebook, or download them from the releases page.
