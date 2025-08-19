# deep-learning-model-for-credit-card-applicant-risk

This project builds an **AI-powered deep learning model** to predict the **credit risk (default probability)** of new credit card applicants.  
It leverages the [UCI Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) to train, evaluate, and serve a predictive model with explainability and monitoring in mind.


## Dataset
- **Name**: Default of Credit Card Clients  
- **Size**: 30,000 instances, 23 features  
- **Target**: `default.payment.next.month` (1 = default, 0 = no default)  
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)  


## Features
- Data preprocessing (missing values, outlier handling, feature engineering)  
- Class imbalance handling (SMOTE, threshold tuning)  
- Deep learning model with probability calibration  
- Evaluation metrics: AUC, Brier Score, Confusion Matrix  
- SHAP-based explainability (global + per-applicant)  
- REST API for real-time scoring (FastAPI)  

