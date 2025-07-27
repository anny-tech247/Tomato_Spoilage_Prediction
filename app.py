import pickle
import numpy as np
from flask import Flask, request, render_template
import logging
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model with error handling
try:
    model_path = 'tomato_spoilage_xgb_model.pkl'
    if not os.path.exists(model_path):
        model_path = 'model/tomato_spoilage_xgb_model.pkl'
    
    model = pickle.load(open(model_path, 'rb'))
    logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('index.html')
    
    # Handle POST request (prediction)
    try:
        # Validate model is loaded
        if model is None:
            return render_template('index.html', 
                                 error="Model not available. Please check server configuration.")
        
        # Get and validate input
        temperature_str = request.form.get('temperature', '').strip()
        humidity_str = request.form.get('humidity', '').strip()
        
        if not temperature_str or not humidity_str:
            return render_template('index.html', 
                                 error="Please fill in all fields.",
                                 temperature=temperature_str,
                                 humidity=humidity_str)
        
        try:
            temperature = float(temperature_str)
            humidity = float(humidity_str)
        except ValueError:
            return render_template('index.html', 
                                 error="Please enter valid numbers.",
                                 temperature=temperature_str,
                                 humidity=humidity_str)
        
        # Validate ranges
        if not (-10 <= temperature <= 50):
            return render_template('index.html', 
                                 error="Temperature must be between -10°C and 50°C.",
                                 temperature=temperature_str,
                                 humidity=humidity_str)
        
        if not (0 <= humidity <= 100):
            return render_template('index.html', 
                                 error="Humidity must be between 0% and 100%.",
                                 temperature=temperature_str,
                                 humidity=humidity_str)
        
        # Calculate features
        temp_x_humidity = temperature * humidity
        features = np.array([[temperature, humidity, temp_x_humidity]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Get probability/confidence if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(features)[0]
                confidence = max(probabilities)  # Highest probability
            except Exception as e:
                logger.warning(f"Could not get prediction probabilities: {str(e)}")
        
        # Format result
        if prediction == 1:
            result = "🚨 The tomato is likely to spoil soon"
        else:
            result = "✅ The tomato is still fresh"
        
        logger.info(f"Prediction made: T={temperature}, H={humidity}, Result={prediction}")
        
        return render_template('index.html', 
                             prediction=result,
                             confidence=confidence,
                             temperature=temperature_str,
                             humidity=humidity_str)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template('index.html', 
                             error=f"An unexpected error occurred: {str(e)}",
                             temperature=request.form.get('temperature', ''),
                             humidity=request.form.get('humidity', ''))

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html', error="Page not found."), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html', error="Internal server error."), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
