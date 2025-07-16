import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import re
import urllib.parse

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

def create_sample_dataset():
    """Create a sample dataset of URLs with labels"""
    
    # Legitimate URLs
    legitimate_urls = [
        "https://www.google.com",
        "https://www.facebook.com",
        "https://www.amazon.com",
        "https://www.microsoft.com",
        "https://www.apple.com",
        "https://www.github.com",
        "https://www.stackoverflow.com",
        "https://www.linkedin.com",
        "https://www.youtube.com",
        "https://www.wikipedia.org",
        "https://www.twitter.com",
        "https://www.instagram.com",
        "https://www.netflix.com",
        "https://www.reddit.com",
        "https://www.ebay.com",
        "https://www.paypal.com",
        "https://www.walmart.com",
        "https://www.target.com",
        "https://www.bestbuy.com",
        "https://www.cnn.com",
        "https://www.bbc.com",
        "https://www.nytimes.com",
        "https://www.adobe.com",
        "https://www.salesforce.com",
        "https://www.oracle.com",
        "https://www.ibm.com",
        "https://www.tesla.com",
        "https://www.spotify.com",
        "https://www.dropbox.com",
        "https://www.zoom.us",
        "https://accounts.google.com/signin",
        "https://login.microsoftonline.com",
        "https://www.amazon.com/gp/signin",
        "https://secure.paypal.com/signin",
        "https://www.facebook.com/login",
        "https://github.com/login",
        "https://stackoverflow.com/users/login",
        "https://www.linkedin.com/login",
        "https://support.google.com",
        "https://help.microsoft.com",
        "https://customer-service.amazon.com",
        "https://www.apple.com/support",
        "https://support.github.com",
        "https://help.paypal.com",
        "https://www.facebook.com/help",
        "https://help.linkedin.com",
        "https://support.spotify.com",
        "https://help.dropbox.com",
        "https://support.zoom.us"
    ]
    
    # Phishing URLs (simulated - these are fake examples for training)
    phishing_urls = [
        "http://secure-paypal-verification.com",
        "https://amazon-security-alert.net",
        "http://google-account-suspended.org",
        "https://facebook-security-check.com",
        "http://microsoft-account-locked.net",
        "https://apple-id-verification.org",
        "http://github-security-notice.com",
        "https://linkedin-account-verify.net",
        "http://paypal-account-limited.org",
        "https://amazon-account-suspend.com",
        "http://192.168.1.1/paypal-login",
        "https://192.168.0.1/secure-banking",
        "http://10.0.0.1/microsoft-login",
        "https://172.16.0.1/google-verify",
        "http://bit.ly/fake-paypal-login",
        "https://tinyurl.com/fake-amazon",
        "http://goo.gl/fake-google-login",
        "https://t.co/fake-twitter-verify",
        "http://secure-bank-update@phishing.com",
        "https://paypal-verify@fake-site.net",
        "http://amazon-security@malicious.org",
        "https://google-account@phishing.com",
        "http://www.paypal-verification-required.com",
        "https://www.amazon-account-verification.net",
        "http://www.google-security-alert.org",
        "https://www.facebook-account-locked.com",
        "http://www.microsoft-account-suspended.net",
        "https://www.apple-id-blocked.org",
        "http://www.github-security-warning.com",
        "https://www.linkedin-account-restricted.net",
        "http://secure-paypal-login.fake-domain.com",
        "https://amazon-customer-service.phishing.net",
        "http://google-account-recovery.malicious.org",
        "https://facebook-help-center.fake-site.com",
        "http://microsoft-support-team.phishing.net",
        "https://apple-customer-support.fake-domain.org",
        "http://github-technical-support.malicious.com",
        "https://linkedin-help-desk.phishing.net",
        "http://paypal-customer-care.fake-site.org",
        "https://amazon-technical-support.malicious.com",
        "http://secure-banking-login.phishing.net/update-account",
        "https://verify-paypal-account.fake-site.com/confirm-identity",
        "http://amazon-security-department.malicious.org/suspend-account",
        "https://google-account-verification.phishing.net/verify-now",
        "http://facebook-security-team.fake-domain.com/account-locked",
        "https://microsoft-account-recovery.malicious.org/reset-password",
        "http://apple-id-support-team.phishing.net/unlock-account",
        "https://github-security-department.fake-site.com/verify-account",
        "http://linkedin-account-verification.malicious.org/confirm-identity",
        "https://paypal-fraud-prevention.phishing.net/secure-login"
    ]
    
    # Create dataset
    urls = legitimate_urls + phishing_urls
    labels = [0] * len(legitimate_urls) + [1] * len(phishing_urls)
    
    return urls, labels

def train_model():
    """Train the phishing detection model"""
    print("Creating sample dataset...")
    urls, labels = create_sample_dataset()
    
    print(f"Total URLs: {len(urls)}")
    print(f"Legitimate URLs: {labels.count(0)}")
    print(f"Phishing URLs: {labels.count(1)}")
    
    # Extract features for all URLs
    print("Extracting features...")
    features_list = []
    for url in urls:
        features = extract_features(url)
        features_list.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    
    # Feature names
    feature_names = df.columns.tolist()
    print(f"Features: {feature_names}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        df, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train the model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    # Save the model and feature names
    print("\nSaving model...")
    joblib.dump(model, 'phishing_model.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')
    
    print("Model saved successfully!")
    
    # Test with sample URLs
    print("\nTesting model with sample URLs:")
    test_urls = [
        "https://www.google.com",
        "http://secure-paypal-verification.com",
        "https://www.amazon.com",
        "http://192.168.1.1/paypal-login"
    ]
    
    for url in test_urls:
        features = extract_features(url)
        features_df = pd.DataFrame([features])
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0]
        result = "PHISHING" if prediction == 1 else "LEGITIMATE"
        confidence = max(probability) * 100
        print(f"{url}: {result} (Confidence: {confidence:.1f}%)")

if __name__ == "__main__":
    train_model()