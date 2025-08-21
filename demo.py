"""
Bank Credit Card Fraud Detection Demo
=====================================

This demo shows how banks can use the real-time fraud detection system
to assess credit card application risk.
"""

import requests
import json
import time
from datetime import datetime

def print_header(title):
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")

def print_application_details(app_data, title):
    print(f"\n{title}:")
    print(f"  Age: {app_data['age']} years")
    print(f"  Credit Limit Requested: ${app_data['limit_bal']:,}")
    print(f"  Education: {['Unknown', 'Graduate School', 'University', 'High School', 'Other'][app_data['education']]}")
    print(f"  Marital Status: {['Unknown', 'Married', 'Single', 'Other'][app_data['marriage']]}")
    print(f"  Recent Payment Status: {app_data['pay_0']} months delay")
    print(f"  Current Bill: ${app_data['bill_amt1']:,}")
    print(f"  Last Payment: ${app_data['pay_amt1']:,}")

def print_risk_assessment(result):
    print(f"\n🤖 AI RISK ASSESSMENT:")
    print(f"  Default Probability: {result['default_probability']:.1%}")
    print(f"  Risk Category: {result['risk_category']}")
    print(f"  Recommendation: {result['recommendation']}")
    print(f"  Confidence Score: {result['confidence_score']:.1%}")
    
    # Risk level indicator
    risk_level = result['default_probability']
    if risk_level < 0.3:
        print(f"  🟢 LOW RISK - Approve immediately")
    elif risk_level < 0.7:
        print(f"  🟡 MEDIUM RISK - Manual review recommended")
    else:
        print(f"  🔴 HIGH RISK - Reject or require additional verification")

def create_demo_applications():
    """Create demo applications representing different risk profiles."""
    
    # Low Risk: Young professional with good payment history
    low_risk = {
        "limit_bal": 80000.0,
        "sex": 1,
        "education": 2,  # University
        "marriage": 2,   # Single
        "age": 28,
        "pay_0": 0,      # No delay
        "pay_2": 0,
        "pay_3": 0,
        "pay_4": 0,
        "pay_5": 0,
        "pay_6": 0,
        "bill_amt1": 5000.0,
        "bill_amt2": 4500.0,
        "bill_amt3": 5200.0,
        "bill_amt4": 4800.0,
        "bill_amt5": 5100.0,
        "bill_amt6": 4900.0,
        "pay_amt1": 5000.0,  # Pays in full
        "pay_amt2": 4500.0,
        "pay_amt3": 5200.0,
        "pay_amt4": 4800.0,
        "pay_amt5": 5100.0,
        "pay_amt6": 4900.0
    }
    
    # Medium Risk: Established customer with occasional delays
    medium_risk = {
        "limit_bal": 45000.0,
        "sex": 2,
        "education": 3,  # High school
        "marriage": 1,   # Married
        "age": 42,
        "pay_0": 1,      # 1 month delay
        "pay_2": 0,
        "pay_3": 2,      # 2 months delay
        "pay_4": 0,
        "pay_5": 1,
        "pay_6": 0,
        "bill_amt1": 12000.0,
        "bill_amt2": 11500.0,
        "bill_amt3": 13000.0,
        "bill_amt4": 10500.0,
        "bill_amt5": 12500.0,
        "bill_amt6": 11000.0,
        "pay_amt1": 500.0,   # Minimum payments
        "pay_amt2": 450.0,
        "pay_amt3": 600.0,
        "pay_amt4": 400.0,
        "pay_amt5": 500.0,
        "pay_amt6": 450.0
    }
    
    # High Risk: Multiple payment delays and high utilization
    high_risk = {
        "limit_bal": 15000.0,
        "sex": 1,
        "education": 4,  # Other
        "marriage": 3,   # Other
        "age": 24,
        "pay_0": 4,      # 4 months delay
        "pay_2": 3,      # 3 months delay
        "pay_3": 5,      # 5 months delay
        "pay_4": 2,
        "pay_5": 4,
        "pay_6": 3,
        "bill_amt1": 18000.0,  # Over limit
        "bill_amt2": 17500.0,
        "bill_amt3": 19000.0,
        "bill_amt4": 16800.0,
        "bill_amt5": 18500.0,
        "bill_amt6": 17200.0,
        "pay_amt1": 0.0,     # No payments
        "pay_amt2": 100.0,   # Minimal payments
        "pay_amt3": 0.0,
        "pay_amt4": 200.0,
        "pay_amt5": 50.0,
        "pay_amt6": 0.0
    }
    
    return [
        ("Professional with Excellent Credit", low_risk),
        ("Established Customer with Mixed History", medium_risk),
        ("Young Adult with Poor Payment History", high_risk)
    ]

def demo_real_time_processing():
    """Demonstrate real-time fraud detection for banks."""
    
    print_header("BANK CREDIT CARD FRAUD DETECTION SYSTEM")
    print("Real-time AI-powered risk assessment for credit applications")
    print(f"Demo Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    
    # API endpoint
    api_url = "http://localhost:8000"
    
    # Check if API is running
    try:
        health_response = requests.get(f"{api_url}/health", timeout=5)
        if health_response.status_code != 200:
            print("❌ Error: API server is not responding properly")
            return
    except requests.ConnectionError:
        print("❌ Error: Cannot connect to API server")
        print("Please start the server with: python api.py")
        return
    
    print("✅ Connected to fraud detection API")
    
    # Get demo applications
    applications = create_demo_applications()
    
    print_header("PROCESSING CREDIT CARD APPLICATIONS")
    
    for i, (description, app_data) in enumerate(applications, 1):
        print(f"\n📋 APPLICATION #{i}")
        print_application_details(app_data, f"APPLICANT: {description}")
        
        # Simulate processing time
        print("\n⏳ Processing application through AI fraud detection...")
        time.sleep(1)
        
        try:
            # Make prediction
            response = requests.post(f"{api_url}/predict", json=app_data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print_risk_assessment(result)
                
                # Show key factors
                print(f"\n📊 KEY RISK FACTORS:")
                for j, factor in enumerate(result['explanation']['top_contributing_features'][:3], 1):
                    print(f"  {j}. {factor['feature']}: {factor['importance']:.3f}")
                
            else:
                print(f"❌ Error: API returned status {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error processing application: {str(e)}")
        
        print("\n" + "-"*60)
    
    # Demonstrate batch processing
    print_header("BATCH PROCESSING DEMONSTRATION")
    print("Processing multiple applications simultaneously...")
    
    try:
        batch_data = [app[1] for app in applications]
        response = requests.post(f"{api_url}/predict/batch", json=batch_data, timeout=15)
        
        if response.status_code == 200:
            batch_result = response.json()
            print(f"✅ Successfully processed {batch_result['count']} applications")
            
            print("\n📈 BATCH RESULTS SUMMARY:")
            for i, pred in enumerate(batch_result['predictions'], 1):
                risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                emoji = risk_emoji.get(pred['risk_category'], "⚪")
                print(f"  Application {i}: {emoji} {pred['risk_category']} Risk "
                      f"({pred['default_probability']:.1%}) - {pred['recommendation']}")
        else:
            print(f"❌ Batch processing failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Batch processing error: {str(e)}")
    
    # Show business impact
    print_header("BUSINESS IMPACT FOR BANKS")
    print("✅ Real-time decision making (< 1 second response)")
    print("✅ Reduced manual review workload")
    print("✅ Consistent risk assessment criteria")
    print("✅ Improved fraud detection accuracy")
    print("✅ Audit trail for compliance")
    print("✅ Scalable to thousands of applications")
    print("\n📞 Contact: Bank IT Department for API integration")

if __name__ == "__main__":
    demo_real_time_processing()