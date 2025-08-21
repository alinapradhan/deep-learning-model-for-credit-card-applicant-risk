"""
Test client for the credit card fraud detection API.
"""
import requests
import json
from typing import Dict, Any

class CreditCardAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def predict_risk(self, application_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict default risk for a single application."""
        response = requests.post(
            f"{self.base_url}/predict",
            json=application_data
        )
        return response.json()
    
    def predict_batch(self, applications: list) -> Dict[str, Any]:
        """Predict default risk for multiple applications."""
        response = requests.post(
            f"{self.base_url}/predict/batch",
            json=applications
        )
        return response.json()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        response = requests.get(f"{self.base_url}/model/info")
        return response.json()

def create_sample_application() -> Dict[str, Any]:
    """Create a sample credit card application for testing."""
    return {
        "limit_bal": 50000.0,
        "sex": 2,  # Female
        "education": 2,  # University
        "marriage": 1,  # Married
        "age": 35,
        "pay_0": 0,  # No delay
        "pay_2": 0,
        "pay_3": 0,
        "pay_4": 0,
        "pay_5": 0,
        "pay_6": 0,
        "bill_amt1": 15000.0,
        "bill_amt2": 12000.0,
        "bill_amt3": 18000.0,
        "bill_amt4": 10000.0,
        "bill_amt5": 8000.0,
        "bill_amt6": 5000.0,
        "pay_amt1": 2000.0,
        "pay_amt2": 1500.0,
        "pay_amt3": 3000.0,
        "pay_amt4": 1000.0,
        "pay_amt5": 800.0,
        "pay_amt6": 500.0
    }

def create_high_risk_application() -> Dict[str, Any]:
    """Create a high-risk application for testing."""
    return {
        "limit_bal": 10000.0,
        "sex": 1,  # Male
        "education": 4,  # Other
        "marriage": 3,  # Other
        "age": 22,
        "pay_0": 3,  # 3 months delay
        "pay_2": 2,  # 2 months delay
        "pay_3": 4,  # 4 months delay
        "pay_4": 2,
        "pay_5": 3,
        "pay_6": 2,
        "bill_amt1": 25000.0,
        "bill_amt2": 30000.0,
        "bill_amt3": 28000.0,
        "bill_amt4": 32000.0,
        "bill_amt5": 35000.0,
        "bill_amt6": 40000.0,
        "pay_amt1": 0.0,  # No payment
        "pay_amt2": 100.0,  # Minimal payment
        "pay_amt3": 0.0,
        "pay_amt4": 200.0,
        "pay_amt5": 0.0,
        "pay_amt6": 150.0
    }

def test_api():
    """Test the API functionality."""
    client = CreditCardAPIClient()
    
    try:
        # Test health check
        print("Testing health check...")
        health = client.health_check()
        print(f"Health status: {health}")
        print()
        
        # Test model info
        print("Testing model info...")
        model_info = client.get_model_info()
        print(f"Model info: {json.dumps(model_info, indent=2)}")
        print()
        
        # Test single prediction - low risk
        print("Testing single prediction (low risk)...")
        low_risk_app = create_sample_application()
        result = client.predict_risk(low_risk_app)
        print(f"Low risk prediction: {json.dumps(result, indent=2)}")
        print()
        
        # Test single prediction - high risk
        print("Testing single prediction (high risk)...")
        high_risk_app = create_high_risk_application()
        result = client.predict_risk(high_risk_app)
        print(f"High risk prediction: {json.dumps(result, indent=2)}")
        print()
        
        # Test batch prediction
        print("Testing batch prediction...")
        batch_result = client.predict_batch([low_risk_app, high_risk_app])
        print(f"Batch predictions: {batch_result['count']} applications processed")
        for i, pred in enumerate(batch_result['predictions']):
            print(f"  Application {i+1}: {pred['risk_category']} risk ({pred['default_probability']:.3f})")
        print()
        
        print("All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to API. Make sure the server is running.")
        print("Start the server with: python api.py")
    except Exception as e:
        print(f"Test error: {str(e)}")

if __name__ == "__main__":
    test_api()