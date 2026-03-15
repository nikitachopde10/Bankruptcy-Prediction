# Bankruptcy Prediction: Econometric Risk Modeling
### Purdue University Kaggle Competition

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Microsoft-green.svg)](https://lightgbm.readthedocs.io/)

## 📌 Project Overview
This project was developed as part of a **Purdue University Kaggle Competition** to predict corporate bankruptcy using a variety of econometric indicators. The primary challenge involved handling a highly imbalanced dataset where bankruptcy cases (Class 1) were rare compared to stable companies (Class 0).

The objective was to build a robust classification pipeline that maximizes the **Area Under the Curve (AUC)**, ensuring high-risk firms are accurately prioritized.

### 🏆 Competition Results
* **Private Leaderboard AUC:** `0.91191` (Final Selected Model)
* **Public Leaderboard AUC:** `0.90500`
* **Performance Strategy:** Successfully prioritized model generalization, allowing a simpler, well-tuned model to outperform complex ensembles on hidden test data.

---

## ⚙️ The Winning Strategy: LightGBM
While I experimented with sophisticated stacking and blending techniques, the final submission utilized a **Single LightGBM Model**. 

**Key Insight:** During the validation phase, the LightGBM model achieved a peak **Validation AUC of 0.8955** at iteration 1105. While complex ensembles showed higher performance on training data, this single-model approach proved more robust against overfitting and generalized better to the competition's private leaderboard.

### Model Specifications:
* **Algorithm:** Light Gradient Boosting Machine (GBDT)
* **Optimization:** Early stopping (200 round patience) to prevent overfitting.
* **Hyperparameters:**
    * `learning_rate`: 0.01
    * `num_leaves`: 31
    * `feature_fraction`: 0.8
    * `bagging_fraction`: 0.8

---

## 🧪 Experiments & Evolution
I evaluated two distinct strategies to arrive at the final result:

| Strategy | Components | OOF AUC | Private AUC |
| :--- | :--- | :--- | :--- |
| **Final Submission** | Single LightGBM (Tuned) | 0.8955 | **0.91191** |
| **Ensemble Model** | 6-Model Stack (LGBM, CatBoost, XGBoost) | 0.91425 | 0.91181 |

### 1. Feature Engineering & Preprocessing
* **Standardization:** Utilized `StandardScaler` to ensure feature consistency across various boosting algorithms.
* **Imbalance Management:** Implemented stratified train-validation splits to maintain the distribution of the rare bankruptcy class.
* **Refit Strategy:** Once optimal iterations were found via early stopping, the model was retrained on the full dataset to capture maximum information before test set prediction.

### 2. The Ensemble Approach
The ensemble (F26) combined six base learners to improve diversity:
* **Base Models:** Multiple LightGBM variants (Full vs. Top 25 features), CatBoost, and XGBoost.
* **Adversarial Threshold Search:** Used to optimize the F1-score, significantly improving the model's ability to balance precision and recall on the training set (F1 increased from 0.13 to **0.509**).

---

## 📊 Evaluation Metrics
Because the dataset was imbalanced, simple accuracy (95.5%) was not the primary focus. The model was evaluated on:
* **AUC:** To measure the quality of the risk ranking.
* **Confusion Matrix:** Specifically monitoring False Negatives (firms that go bankrupt but were predicted safe).
* **Precision/Recall:** Achieving a high precision (0.78) for the minority class in the final LightGBM validation.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** `LightGBM`, `XGBoost`, `CatBoost`, `Scikit-Learn`, `Pandas`, `NumPy`
* **Workflow:** Data Cleaning -> Feature Scaling -> Hyperparameter Tuning -> Model Stacking -> Final Refit.

---

## 🏁 Conclusion
This project highlights the importance of the **Bias-Variance tradeoff**. While the ensemble was more complex and performed better on training metrics, the simpler LightGBM model provided the stability and generalization necessary to win on the private leaderboard.
