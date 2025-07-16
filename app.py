from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import re
import urllib.parse
import os

app = Flask(__name__)

# Load the trained model and feature names
MODEL_PATH = 'phishing_model.pkl'
FEATURES_PATH = 'feature_names.pkl'

def load_model():
    """Load the trained model and feature names"""
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        return model, feature_names
    else:
        raise FileNotFoundError("Model files not found. Please run train_model.py first.")

# Load model at startup
try:
    model, feature_names = load_model()
    print("Model loaded successfully!")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please run 'python train_model.py' first to train the model.")
    model, feature_names = None, None

def extract_features(url):
    """Extract features from a URL for phishing detection"""
    features = {}
    
    # Basic URL features
    features['url_length'] = len(url)
    features['num_dots'] = url.count('.')
    features['has_at'] = 1 if '@' in url else 0
    features['has_hyphen'] = 1 if '-' in urllib.parse.urlparse(url).netloc else 0
    
    # Check if URL uses IP address instead of domain
    parsed = urllib.parse.urlparse(url)
    ip_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    features['has_ip'] = 1 if re.match(ip_pattern, parsed.netloc.split(':')[0]) else 0
    
    # Path length
    features['path_length'] = len(parsed.path)
    
    # Number of subdomains
    domain_parts = parsed.netloc.split('.')
    features['num_subdomains'] = len(domain_parts) - 2 if len(domain_parts) > 2 else 0
    
    # Suspicious characters count
    suspicious_chars = ['%', '&', '=', '?', '#']
    features['suspicious_chars'] = sum(url.count(char) for char in suspicious_chars)
    
    # Check for suspicious keywords
    suspicious_keywords = ['secure', 'account', 'update', 'confirm', 'login', 'verify', 'suspend']
    features['suspicious_keywords'] = sum(1 for keyword in suspicious_keywords if keyword in url.lower())
    
    # URL entropy (measure of randomness)
    def calculate_entropy(s):
        prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
        entropy = -sum([p * np.log2(p) for p in prob])
        return entropy
    
    features['url_entropy'] = calculate_entropy(url)
    
    return features

def predict_phishing(url):
    """Predict if a URL is phishing or legitimate"""
    if model is None:
        return None, None, None
    
    try:
        # Extract features
        features = extract_features(url)
        
        # Create DataFrame with the same column order as training
        features_df = pd.DataFrame([features])
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0]
        
        # Get result and confidence
        result = "PHISHING" if prediction == 1 else "LEGITIMATE"
        confidence = max(probability) * 100
        
        return result, confidence, features
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None, None, None

@app.route('/', methods=['GET', 'POST'])
def index():
    """Main page with URL input form"""
    if request.method == 'POST':
        url = request.form.get('url', '').strip()
        
        if not url:
            return render_template('index.html', error="Please enter a URL")
        
        # Add http:// if no protocol specified
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # Predict
        result, confidence, features = predict_phishing(url)
        
        if result is None:
            return render_template('index.html', 
                                 error="Model not available. Please train the model first.",
                                 url=url)
        
        return render_template('index.html', 
                             url=url,
                             result=result,
                             confidence=f"{confidence:.1f}",
                             features=features)
    
    return render_template('index.html')

@app.route('/api/check', methods=['POST'])
def api_check():
    """API endpoint for URL checking"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Add http:// if no protocol specified
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        # Predict
        result, confidence, features = predict_phishing(url)
        
        if result is None:
            return jsonify({'error': 'Model not available'}), 500
        
        return jsonify({
            'url': url,
            'result': result,
            'confidence': f"{confidence:.1f}",
            'features': features
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)