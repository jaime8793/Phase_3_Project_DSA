# H1N1 Vaccine Prediction: Optimizing Recall for Public Health

## Project Overview
This project builds a predictive machine learning model to identify individuals most likely to receive the H1N1 flu vaccine. Using data from the National 2009 H1N1 Flu Survey, the goal is to guide public health efforts by identifying key drivers of vaccination and predicting positive cases with high sensitivity.

**Key Achievement:** optimized the model to increase **Recall to ~62%** (catching the majority of vaccinated individuals) while maintaining a balanced Precision, effectively solving the "blindness" of the baseline model towards the minority class.

## Business Problem
Vaccination rates for H1N1 are low (~20% of the population). Public health campaigns have limited resources and cannot target everyone.
* **Goal:** Predict who is likely to get vaccinated to better target educational campaigns and provider outreach.
* **Challenge:** The dataset is highly imbalanced (80% unvaccinated / 20% vaccinated). A standard model tends to predict "No" for everyone to achieve high accuracy, which is useless for identifying targets.

## Tech Stack & Methodology
* **Language:** Python 3.9+
* **Libraries:** `scikit-learn`, `xgboost`, `pandas`, `seaborn`, `matplotlib`, `category_encoders`
* **Model:** XGBoost Classifier (Extreme Gradient Boosting)

### The Pipeline
1.  **Preprocessing:**
    * **Numerical:** Missing values filled via `IterativeImputer` (MICE), followed by `StandardScaler`.
    * **High-Cardinality Categorical:** Processed using `CountEncoder` (Frequency Encoding) to capture category importance without exploding dimensions.
    * **Low-Cardinality Categorical:** Processed using `OneHotEncoder`.
2.  **Handling Imbalance:**
    * Used **Cost-Sensitive Learning** (`scale_pos_weight=3`) rather than SMOTE. This penalized the model 3x more for missing a positive case, forcing it to pay attention to the minority class.
3.  **Optimization:**
    * Hyperparameter tuning via `GridSearchCV`.
    * **Decision Threshold Tuning:** Shifted the classification threshold from 0.50 to **0.60** to maximize the F1-Score and balance Precision/Recall.

##  Model Performance
Comparison between the Baseline (Standard) and the Final Optimized Model:

| Metric | Baseline Model | Optimized Model (Threshold 0.60) | Impact |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | 0.73 | **0.86** |  Drastic improvement in discriminative power |
| **Recall** | ~52% | **62%** |  Caught +10% more positive cases |
| **Precision** | ~72% | **61%** |  Accepted trade-off to reduce missed opportunities |
| **Accuracy** | ~85% | **84%** |  Negligible drop |

**Final Verdict:** The optimized model is robust and unbiased, successfully balancing the need to catch cases (Recall) with the need to be accurate (Precision).

##  Key Insights & Drivers
Feature Importance analysis using XGBoost revealed the following hierarchy of influence:

1.  **Doctor's Recommendation is King:** The feature `doctor_recc_h1n1` was nearly **3x more influential** than any other factor. If a doctor recommends the shot, the patient is highly likely to comply.
2.  **Fear Over Logic:** `opinion_h1n1_risk` (Perception of Risk) outweighed `opinion_h1n1_vacc_effective` (Belief in Effectiveness). Patients are more motivated by the fear of getting sick than the logic of the cure.
3.  **Access is Secondary:** `health_insurance` was a top driver, but less significant than the psychological and authority-based factors above.

##  Visualizations
* **Confusion Matrix:** Shows the balance between True Positives (709) and False Negatives (426).
* **ROC Curve:** Demonstrates the model's strong ability to rank positive cases higher than negative ones (AUC = 0.86).
* **Feature Importance Plot:** Visual confirmation of the "Doctor Recommendation" dominance.

##  Recommendations
Based on the modeling results, the following strategy is recommended:
1.  **Prioritize Provider Outreach:** Shift marketing budget from general awareness ads to equipping doctors with reminder scripts, as this is the strongest lever for conversion.
2.  **Target "Risk" Messaging:** For non-doctor-visiting populations, focus campaigns on the *susceptibility* and *severity* of the flu, rather than just vaccine safety.
3.  **Deploy at Threshold 0.60:** Implement the model with a strict 0.60 decision threshold to filter out weak false positives while retaining high detection rates.

##  Installation
1. Clone the repo:
   ```bash
   git clone [https://github.com/jaime8793/Phase_3_Project_DSA](https://github.com/jaime8793/Phase_3_Project_DSA)