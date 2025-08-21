# Credit Card Fraud Detection System

This project implements a **real-time credit card fraud detection system** for banks using deep learning. The system predicts the **credit risk (default probability)** of new credit card applicants and provides real-time risk assessment through a REST API.

## 🎯 Features

- **Deep Learning Model**: TensorFlow/Keras neural network for credit risk prediction
- **Real-time API**: FastAPI-based REST API for instant fraud detection
- **Data Preprocessing**: Automated feature engineering and scaling
- **Class Imbalance Handling**: SMOTE for balanced training
- **Model Explainability**: Feature importance analysis
- **Risk Categorization**: Low/Medium/High risk classification
- **Batch Processing**: Support for multiple applications
- **Production Ready**: Comprehensive logging and error handling

## 📊 Dataset

- **Name**: Default of Credit Card Clients  
- **Size**: 30,000 instances, 23 features  
- **Target**: `default.payment.next.month` (1 = default, 0 = no default)  
- **Source**: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)  

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/alinapradhan/deep-learning-model-for-credit-card-applicant-risk.git
cd deep-learning-model-for-credit-card-applicant-risk

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Train the fraud detection model
python train.py
```

This will:
- Preprocess the UCI Credit Card dataset
- Train a deep neural network
- Save the trained model and preprocessor
- Generate evaluation plots

### 3. Start the API Server

```bash
# Start the FastAPI server
python api.py
```

The API will be available at `http://localhost:8000`

### 4. Test the System

```bash
# Test the API with sample data
python test_api.py
```

## 📝 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "limit_bal": 50000,
       "sex": 2,
       "education": 2,
       "marriage": 1,
       "age": 35,
       "pay_0": 0,
       "pay_2": 0,
       "pay_3": 0,
       "pay_4": 0,
       "pay_5": 0,
       "pay_6": 0,
       "bill_amt1": 15000,
       "bill_amt2": 12000,
       "bill_amt3": 18000,
       "bill_amt4": 10000,
       "bill_amt5": 8000,
       "bill_amt6": 5000,
       "pay_amt1": 2000,
       "pay_amt2": 1500,
       "pay_amt3": 3000,
       "pay_amt4": 1000,
       "pay_amt5": 800,
       "pay_amt6": 500
     }'
```

### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

## 🏗️ Architecture

```
├── data_preprocessing.py  # Data cleaning and feature engineering
├── model.py              # Deep learning model implementation
├── api.py               # FastAPI REST API
├── train.py             # Model training script
├── test_api.py          # API testing client
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
└── models/              # Saved models and artifacts
```

## 🔧 Configuration

Key parameters can be configured in `config.py`:

- **Model Architecture**: Hidden layers, dropout rates, activation functions
- **Training Settings**: Epochs, batch size, learning rate
- **Risk Thresholds**: Low/Medium/High risk boundaries
- **API Settings**: Host, port, title, version

## 📈 Model Performance

The model uses the following architecture:
- **Input Layer**: 23 features
- **Hidden Layers**: 128 → 64 → 32 → 16 neurons with ReLU activation
- **Regularization**: Batch normalization and dropout
- **Output**: Single probability score (0-1)

**Evaluation Metrics**:
- AUC Score
- Precision/Recall
- Confusion Matrix
- ROC Curve

## 🏦 Banking Use Cases

### Real-time Application Processing
- Instant credit card application assessment
- Risk-based decision making
- Automated approval/rejection

### Fraud Prevention
- Early detection of high-risk applicants
- Reduced financial losses
- Improved portfolio quality

### Compliance and Reporting
- Audit trail for decisions
- Risk category tracking
- Performance monitoring

## 📊 API Response Format

```json
{
  "default_probability": 0.156,
  "risk_category": "Low",
  "recommendation": "Approve",
  "confidence_score": 0.688,
  "explanation": {
    "top_contributing_features": [
      {"feature": "PAY_0", "importance": 0.234},
      {"feature": "LIMIT_BAL", "importance": 0.187}
    ],
    "input_summary": {
      "age": 35,
      "credit_limit": 50000,
      "education_level": 2,
      "recent_payment_status": 0
    }
  }
}
```

## 🔒 Security Considerations

- Input validation for all API endpoints
- Rate limiting for production deployment
- Secure model artifact storage
- Audit logging for all predictions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

