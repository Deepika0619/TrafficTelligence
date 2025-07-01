from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and preprocessing tools
model = joblib.load('model.pkl')
holiday_encoder = joblib.load('holiday_encoder.pkl')
weather_encoder = joblib.load('weather_encoder.pkl')
scaler = joblib.load('scale.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs and clean them
        holiday = request.form['holiday'].strip().lower()
        weather = request.form['weather'].strip().lower()
        temp = request.form['temp'].strip()
        rain = request.form['rain'].strip()
        snow = request.form['snow'].strip()
        year = request.form['year'].strip()
        month = request.form['month'].strip()
        day = request.form['day'].strip()
        hour = request.form['hour'].strip()
        minutes = request.form['minutes'].strip()
        seconds = request.form['seconds'].strip()

        # Check for missing fields
        if not all([holiday, weather, temp, rain, snow, year, month, day, hour, minutes, seconds]):
            return render_template('index.html', error="Please fill all fields.")

        # Convert numeric values
        temp = float(temp)
        rain = float(rain)
        snow = float(snow)
        year = int(year)
        month = int(month)
        day = int(day)
        hour = int(hour)
        minutes = int(minutes)
        seconds = int(seconds)

        # Handle unseen categories
        if holiday not in holiday_encoder.classes_:
            holiday = 'none'
        if weather not in weather_encoder.classes_:
            weather = 'clear'

        # Encode
        holiday_encoded = holiday_encoder.transform([holiday])[0]
        weather_encoded = weather_encoder.transform([weather])[0]

        # Combine all input features
        input_features = [[holiday_encoded, temp, rain, snow, weather_encoded,
                           year, month, day, hour, minutes, seconds]]

        # Scale inputs
        input_scaled = scaler.transform(input_features)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction = round(prediction)

        # Render result
        if prediction > 4000:
            return render_template('chance.html', prediction=prediction)
        else:
            return render_template('noChance.html', prediction=prediction)

    except Exception as e:
        import traceback
        traceback.print_exc()  # 👈 Add this line to print full error
        print("Prediction error:", e)
        return render_template('index.html', error="Something went wrong.")


if __name__ == '__main__':
    app.run(debug=True)
