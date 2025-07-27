import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Use correct path depending on your choice:
model = pickle.load(open('tomato_spoilage_xgb_model.pkl', 'rb'))  # if in root
# model = pickle.load(open('model/tomato_spoilage_xgb_model.pkl', 'rb'))  # if inside model/

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        
        temp_x_humidity = temperature * humidity

        # Prepare input in the same order you trained
        features = np.array([[temperature, humidity, temp_x_humidity]])
        
        prediction = model.predict(features)[0]
        
        if prediction == 1:
            result = "The tomato is likely to spoil soon (1)."
        else:
            result = "The tomato is still fresh (0)."

        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
