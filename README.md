# ClickSentry - Phishing URL Detection

A modern web application that uses machine learning to detect phishing URLs in real-time. Built with Python, Flask, and scikit-learn.

## 🚀 Features

- **Real-time URL Analysis**: Instantly analyze URLs for phishing indicators
- **Machine Learning Detection**: Uses Random Forest algorithm with multiple URL features
- **Beautiful Web Interface**: Modern, responsive design with Bootstrap
- **Detailed Analysis**: Shows feature breakdown and confidence scores
- **API Endpoint**: RESTful API for integration with other applications
- **Local Deployment**: Easy to run locally for testing and development

## 🛠️ Tech Stack

- **Backend**: Python 3.8+, Flask
- **Machine Learning**: scikit-learn, pandas, numpy
- **Frontend**: HTML5, CSS3, Bootstrap 5, JavaScript
- **Model**: Random Forest Classifier

## 📁 Project Structure

```
ClickSentry/
├── app.py                 # Flask web application
├── train_model.py         # Model training script
├── setup.py              # Automated setup script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── templates/
│   └── index.html       # Main web interface
├── phishing_model.pkl   # Trained model (generated)
└── feature_names.pkl    # Feature names (generated)
```

## 🔧 Installation & Setup

### Method 1: Automated Setup (Recommended)

1. **Clone or download the project**
2. **Open Command Prompt or PowerShell**
3. **Navigate to the project directory**:
   ```bash
   cd c:/Users/ASUS/Desktop/ClickSentry
   ```
4. **Run the setup script**:
   ```bash
   python setup.py
   ```

This will automatically:
- Install all required dependencies
- Train the machine learning model
- Verify the installation

### Method 2: Manual Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the machine learning model**:
   ```bash
   python train_model.py
   ```

3. **Start the web application**:
   ```bash
   python app.py
   ```

## 🚀 Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and go to:
   ```
   http://localhost:5000
   ```

3. **Test the application**:
   - Enter a URL in the text box
   - Click "Check URL"
   - View the results and analysis

## 🧪 Testing

### Sample URLs for Testing

**Legitimate URLs** (should be classified as safe):
- https://www.google.com
- https://www.amazon.com
- https://github.com/login

**Suspicious URLs** (should be classified as phishing):
- http://secure-paypal-verification.com
- https://amazon-security-alert.net
- http://192.168.1.1/paypal-login

## 🔍 How It Works

### Feature Extraction

The system analyzes URLs using these features:

1. **URL Length**: Longer URLs are often suspicious
2. **Number of Dots**: Multiple subdomains can indicate phishing
3. **Contains "@" Symbol**: Often used to mislead users
4. **Domain Contains Hyphen**: Suspicious in domain names
5. **Uses IP Address**: Legitimate sites rarely use IP addresses
6. **Path Length**: Very long paths can be suspicious
7. **Number of Subdomains**: Many subdomains are suspicious
8. **Suspicious Characters**: Count of special characters
9. **Suspicious Keywords**: Presence of phishing-related words
10. **URL Entropy**: Measure of randomness in the URL

### Machine Learning Model

- **Algorithm**: Random Forest Classifier
- **Training Data**: 100 sample URLs (50 legitimate, 50 phishing)
- **Features**: 10 extracted features per URL
- **Accuracy**: Typically 85-95% on test data

## 🌐 API Usage

### Check URL via API

**Endpoint**: `POST /api/check`

**Request**:
```json
{
  "url": "https://example.com"
}
```

**Response**:
```json
{
  "url": "https://example.com",
  "result": "LEGITIMATE",
  "confidence": "92.5",
  "features": {
    "url_length": 19,
    "num_dots": 1,
    "has_at": 0,
    "has_hyphen": 0,
    "has_ip": 0,
    "path_length": 1
  }
}
```

### Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🚀 Deployment Options

### Local Development
Already covered in the setup instructions above.

### Heroku Deployment

1. **Create a Heroku app**:
   ```bash
   heroku create your-app-name
   ```

2. **Create a Procfile**:
   ```
   web: python app.py
   ```

3. **Deploy**:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push heroku main
   ```

### Docker Deployment

1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.9-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   RUN python train_model.py
   EXPOSE 5000
   CMD ["python", "app.py"]
   ```

2. **Build and run**:
   ```bash
   docker build -t clicksentry .
   docker run -p 5000:5000 clicksentry
   ```

## 🔧 Configuration

### Environment Variables

- `FLASK_ENV`: Set to `development` for debugging
- `FLASK_PORT`: Port number (default: 5000)
- `FLASK_HOST`: Host address (default: 0.0.0.0)

### Model Configuration

Edit `train_model.py` to modify:
- Training data
- Model parameters
- Feature extraction logic

## 🎨 Customization Ideas

### UI Improvements
- Add dark mode toggle
- Implement bulk URL checking
- Add URL history/bookmarking
- Create mobile app version

### Model Improvements
- Add more training data
- Implement ensemble methods
- Add real-time learning
- Include domain reputation data

### Features
- Browser extension
- Email integration
- API rate limiting
- User authentication
- Database storage

## 🐛 Troubleshooting

### Common Issues

1. **Model not found error**:
   - Run `python train_model.py` first
   - Check if `.pkl` files exist

2. **Dependencies not installed**:
   - Run `pip install -r requirements.txt`
   - Use virtual environment

3. **Port already in use**:
   - Change port in `app.py`
   - Or kill existing processes

4. **Import errors**:
   - Check Python version (3.8+)
   - Verify all packages installed

## 📊 Performance

- **Model Training Time**: ~2-5 seconds
- **Prediction Time**: <100ms per URL
- **Memory Usage**: ~50MB
- **Accuracy**: 85-95% on test data

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📧 Support

For issues and questions:
- Check the troubleshooting section
- Review the code comments
- Create an issue in the repository

## 🔮 Future Enhancements

- [ ] Real-time URL monitoring
- [ ] Integration with threat intelligence feeds
- [ ] Advanced deep learning models
- [ ] Browser extension development
- [ ] Mobile application
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] API authentication and rate limiting

---

**Happy Phishing Detection! 🛡️**