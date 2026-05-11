# Drowsiness Detection Using Facial Landmark Analysis

A machine learning pipeline for real-time drowsiness detection based on facial landmark metrics. The system extracts Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) features from video frames using MediaPipe and MTCNN, then classifies facial states as **blinking**, **yawning**, or **normal** using multiple ML classifiers.

---

## Overview

This project implements an end-to-end drowsiness detection pipeline:

1. **Video Frame Extraction** — Extracts frames from video recordings at configurable intervals using OpenCV
2. **Face Detection & Cropping** — Detects and crops faces to 224×224 using MTCNN
3. **Automatic Labeling** — Labels frames as `blinking`, `yawning`, or `normal` based on EAR/MAR thresholds via MediaPipe FaceMesh
4. **Data Balancing** — Applies undersampling and data augmentation to handle class imbalance
5. **Feature Engineering** — Computes derived features (EAR−MAR, EAR/MAR, MAR/EAR, EAR×MAR) with MinMaxScaler normalization
6. **Model Training** — Trains and evaluates four classifiers:
   - Support Vector Machine (SVM)
   - Gaussian Naive Bayes
   - XGBoost (with GridSearchCV hyperparameter tuning)
   - Logistic Regression
7. **Evaluation** — Generates classification reports, confusion matrices, and ROC-AUC scores
8. **Inference** — Single-image prediction using trained models

---

## Project Structure

```
drowsiness-detection/
├── notebooks/
│   └── drowsiness_detection_pipeline.ipynb   # Main Jupyter/Colab notebook
├── models/                                    # Trained model files (.pkl)
├── data/                                      # Input data (videos, frames, CSV)
├── results/                                   # Evaluation outputs & visualizations
├── requirements.txt                           # Python dependencies
├── .gitignore                                 # Git ignore rules
└── README.md                                  # This file
```

---

## Key Metrics & Features

| Feature           | Description                                    |
|--------------------|------------------------------------------------|
| `ear`              | Eye Aspect Ratio — detects blinking            |
| `mar`              | Mouth Aspect Ratio — detects yawning           |
| `ear_minus_mar`    | EAR − MAR                                      |
| `ear_div_mar`      | EAR / MAR                                      |
| `mar_div_ear`      | MAR / EAR                                      |
| `ear_times_mar`    | EAR × MAR                                      |

### Thresholds

| Parameter                 | Value |
|---------------------------|-------|
| EAR Threshold (blinking)  | 0.15  |
| MAR Threshold (yawning)   | 0.40  |
| Mouth Closed Threshold    | 0.20  |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Google Colab (recommended) or local Jupyter environment
- GPU recommended for MediaPipe and MTCNN processing

### Installation

```bash
pip install -r requirements.txt
```

### Usage

1. **Open the notebook** in Google Colab or Jupyter Lab:
   ```
   notebooks/drowsiness_detection_pipeline.ipynb
   ```

2. **Prepare your data:**
   - Place video files in the `data/` directory
   - The notebook will extract frames, detect faces, and generate labels automatically

3. **Train models:**
   - Run the notebook cells sequentially to train SVM, Naive Bayes, XGBoost, and Logistic Regression classifiers

4. **Evaluate:**
   - View classification reports, confusion matrices, and ROC-AUC curves in the evaluation section

5. **Inference:**
   - Use the final notebook cells to predict drowsiness state on new images

---

## Models Trained

| Model              | Description                                      |
|---------------------|--------------------------------------------------|
| SVM (RBF kernel)    | `C=1, gamma='scale'`                             |
| Gaussian Naive Bayes| `var_smoothing=1e-9`                              |
| XGBoost             | GridSearchCV tuned (n_estimators, lr, max_depth)  |
| Logistic Regression | `max_iter=1000`                                   |

Trained models are saved as `.pkl` files in the `models/` directory.

---

## Technologies Used

- **OpenCV** — Video processing & frame extraction
- **MTCNN** — Face detection and bounding box extraction
- **MediaPipe** — Facial landmark detection (468 landmarks)
- **scikit-learn** — Model training, evaluation, preprocessing
- **XGBoost** — Gradient boosting classifier
- **imbalanced-learn** — SMOTE for class balancing
- **Matplotlib** — Visualization

---

## License

This project is for academic/research purposes.

---

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) by Google for facial landmark detection
- [MTCNN](https://github.com/ipazc/mtcnn) for face detection
- [XGBoost](https://xgboost.readthedocs.io/) for gradient boosting
