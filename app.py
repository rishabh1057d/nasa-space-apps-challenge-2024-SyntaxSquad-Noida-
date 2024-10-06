# app.py
from flask import Flask, render_template, request, jsonify, session
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from datetime import datetime, timezone
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import random

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management

# Load the pre-trained model and tokenizer from Hugging Face
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Meteomatics API credentials
username = 'singh_rishabh'
password = 'F8lGB4j5yc'

# Define the endpoint and parameters
base_url = 'https://api.meteomatics.com'

# Load your historical weather dataset
data = pd.read_csv('datasets_1.csv', delimiter=',', header=0)
data.columns = data.columns.str.strip()
data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

# Get the last date in the dataset
last_date = data['Date'].max()

# Get the current date
current_date = datetime.now(timezone.utc).date()

# Generate a list of missing dates
missing_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=current_date)

def fetch_weather_data(date):
    formatted_date = date.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # Update parameters to include only the required data
    parameters = (
        'precip_24h:mm,'  # Precipitation for the last 24 hours
        'msl_pressure:hPa,'  # Mean sea level pressure
        'wind_speed_10m:ms,'  # Wind speed at 10m
        'wind_dir_10m:d,'  # Wind direction at 10m
        't_2m:C'  # Temperature at 2m in Celsius
    )
    
    location = '28.5617,77.4834'  # Replace with your actual location
    url = f'{base_url}/{formatted_date}/{parameters}/{location}/json'
    response = requests.get(url, auth=(username, password))
    
    if response.status_code == 200:
        meteomatics_data = response.json()
        
        # Check if 'data' is not empty and has the expected structure
        if 'data' in meteomatics_data and len(meteomatics_data['data']) > 0:
            return {
                'Year': date.year,
                'Date': date.strftime('%d-%m-%Y'),
                'Day of Year': date.dayofyear,
                'Precipitation': meteomatics_data['data'][0]['coordinates'][0]['dates'][0]['value'] if len(meteomatics_data['data'][0]['coordinates']) > 0 else None,
                'Surface Pressure': meteomatics_data['data'][1]['coordinates'][0]['dates'][0]['value'] if len(meteomatics_data['data'][1]['coordinates']) > 0 else None,
                'Wind Speed': meteomatics_data['data'][2]['coordinates'][0]['dates'][0]['value'] if len(meteomatics_data['data'][2]['coordinates']) > 0 else None,
                'Wind Direction': meteomatics_data['data'][3]['coordinates'][0]['dates'][0]['value'] if len(meteomatics_data['data'][3]['coordinates']) > 0 else None,
                'Temperature': meteomatics_data['data'][4]['coordinates'][0]['dates'][0]['value'] if len(meteomatics_data['data'][4]['coordinates']) > 0 else None,
            }
        else:
            print(f"No data returned for {date}.")
            return None
    else:
        print(f"Error fetching data for {date}: {response.status_code} - {response.text}")
        return None

# Create a list to hold new data
new_data = []

for date in missing_dates:
    weather_data = fetch_weather_data(date)
    if weather_data:
        new_data.append(weather_data)

# Convert the new data to a DataFrame
new_data_df = pd.DataFrame(new_data)

# Append the new data to the existing DataFrame
data = pd.concat([data, new_data_df], ignore_index=True)

# Data preprocessing for machine learning
data['Precipitation'] = pd.to_numeric(data['Precipitation'], errors='coerce')
data['Surface Pressure'] = pd.to_numeric(data['Surface Pressure'], errors='coerce')
data['Wind Speed'] = pd.to_numeric(data['Wind Speed'], errors='coerce')
data['Wind Direction'] = pd.to_numeric(data['Wind Direction'], errors='coerce')
data['Temperature'] = pd.to_numeric(data['Temperature'], errors='coerce')

# Impute missing values
imputer = SimpleImputer(strategy='mean')
data[['Precipitation', 'Surface Pressure', 'Wind Speed', 'Wind Direction', 'Temperature']] = imputer.fit_transform(
    data[['Precipitation', 'Surface Pressure', 'Wind Speed', 'Wind Direction', 'Temperature']]
)

# Prepare features and target variables
X = data[['Surface Pressure', 'Wind Speed', 'Wind Direction']]
y_rain = (data['Precipitation'] > 0).astype(int)  # Binary target for rain prediction
y_precipitation = data['Precipitation']

# Split the data into training and testing sets
X_train, X_test, y_train_rain, y_test_rain = train_test_split(X, y_rain, test_size=0.2, random_state=42)
X_train_precip, X_test_precip, y_train_precipitation, y_test_precipitation = train_test_split(X, y_precipitation, test_size=0.2, random_state=42)

# Train Random Forest Classifier for rain prediction
rain_model = RandomForestClassifier(random_state=42)
rain_model.fit(X_train, y_train_rain)

# Train Random Forest Regressor for precipitation prediction
precip_model = RandomForestRegressor(random_state=42)
precip_model.fit(X_train_precip, y_train_precipitation)

# Function to predict weather using the models
def predict_weather(features):
    features_df = pd.DataFrame([features], columns=['Surface Pressure', 'Wind Speed', 'Wind Direction'])
    
    will_rain = rain_model.predict(features_df)[0]
    predicted_precipitation = precip_model.predict(features_df)[0]
    
    # If no rain is expected, set predicted precipitation to 0
    if will_rain == 0:
        predicted_precipitation = 0.0
    
    return will_rain, predicted_precipitation

# Function to retrieve irrigation information
def retrieve_irrigation_info(soil_type, crop_type):
    irrigation_knowledge_base = {
        ('Sandy', 'Wheat'): "Irrigate 15mm per week.",
        ('Loamy', 'Wheat'): "Irrigate 10mm per week.",
        ('Clay', 'Wheat'): "Irrigate 8mm per week.",
        ('Silty', 'Wheat'): "Irrigate 12mm per week.",
        ('Sandy', 'Rice'): "Irrigate 25mm per week.",
        ('Loamy', 'Rice'): "Irrigate 20mm per week.",
        ('Clay', 'Rice'): "Irrigate 18mm per week.",
        ('Silty', 'Rice'): "Irrigate 22mm per week.",
        ('Sandy', 'Corn'): "Irrigate 15mm per week.",
        ('Loamy', 'Corn'): "Irrigate 10mm per week.",
        ('Clay', 'Corn'): "Irrigate 8mm per week.",
        ('Silty', 'Corn'): "Irrigate 12mm per week.",
        ('Sandy', 'Soybean'): "Irrigate 15mm per week.",
        ('Loamy', 'Soybean'): "Irrigate 10mm per week.",
        ('Clay', 'Soybean'): "Irrigate 8mm per week.",
        ('Silty', 'Soybean'): "Irrigate 12mm per week.",
    }
    return irrigation_knowledge_base.get((soil_type, crop_type), None)

# Function to generate irrigation recommendations
def generate_recommendation(soil_type, crop_type, will_rain, predicted_precipitation):
    irrigation_info = retrieve_irrigation_info(soil_type, crop_type)
    if irrigation_info:
        if will_rain:
            recommendation = f"🌧️ Rain is expected. Consider reducing irrigation. {irrigation_info}"
        else:
            recommendation = f"☀️ No rain expected. {irrigation_info}"
    else:
        recommendation = "❌ Invalid soil or crop type. Please select from the given options."
    return recommendation

# Function to validate soil and crop types
def validate_input(soil_type, crop_type):
    valid_soil_types = ['Sandy', 'Loamy', 'Clay', 'Silty']
    valid_crop_types = ['Wheat', 'Rice', 'Corn', 'Soybean']
    
    if soil_type not in valid_soil_types:
        return False, f"Please select a valid soil type: {', '.join(valid_soil_types)}."
    if crop_type not in valid_crop_types:
        return False, f"Please select a valid crop type: {', '.join(valid_crop_types)}."
    
    return True, ""

# Function to generate dynamic responses
def generate_dynamic_response(user_input):
    responses = [
        "That's interesting! Can you tell me more?",
        "I see! What else would you like to know?",
        "That's a great question! Let me think...",
        "Hmm, I'm not sure about that. Can you clarify?",
        "I can help with that! What specific information are you looking for?"
    ]
    return random.choice(responses)

# Chatbot functionality
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input'].strip().lower()  # Normalize input

    # Initialize chat history in session
    if 'soil_type' not in session:
        session['soil_type'] = None
    if 'crop_type' not in session:
        session['crop_type'] = None

    # Check for specific topics
    if "hello" in user_input or "hi" in user_input:
        response = "Hello! How can I assist you today?"
    
    elif "weather" in user_input or "rain" in user_input:
        features = [data['Surface Pressure'].iloc[-1], data['Wind Speed'].iloc[-1], data['Wind Direction'].iloc[-1]]
        will_rain, predicted_precipitation = predict_weather(features)
        current_weather = data.iloc[-1]  # Get the latest weather data
        temperature = current_weather['Temperature']
        surface_pressure = current_weather['Surface Pressure']
        wind_speed = current_weather['Wind Speed']
        wind_direction = current_weather['Wind Direction']
        
        response = (
            f"🌤️ Current Weather:\n"
            f"Temperature: {temperature:.2f} °C\n"
            f"Surface Pressure: {surface_pressure:.2f} hPa\n"
            f"Wind Speed: {wind_speed:.2f} m/s\n"
            f"Wind Direction: {wind_direction:.2f}°\n"
            f"Predicted Precipitation: {predicted_precipitation:.2f} mm\n"
        )
        if will_rain:
            response += "🌧️ Rain is expected."
        else:
            response += "☀️ No rain is expected."
    
    elif "irrigation" in user_input:
        if session['soil_type'] is None:
            response = "Please provide the soil type (Sandy, Loamy, Clay, Silty)."
        else:
            response = "Thank you! Now, please provide the crop type (Wheat, Rice, Corn, Soybean)."
    
    elif user_input in ["sandy", "loamy", "clay", "silty"]:
        session['soil_type'] = user_input.capitalize()  # Store the soil type
        response = "Thank you! Now, please provide the crop type (Wheat, Rice, Corn, Soybean)."
    
    elif user_input in ["wheat", "rice", "corn", "soybean"]:
        if session['soil_type'] is None:
            response = "Please provide the soil type first (Sandy, Loamy, Clay, Silty)."
        else:
            session['crop_type'] = user_input.capitalize()  # Store the crop type
            features = [data['Surface Pressure'].iloc[-1], data['Wind Speed'].iloc[-1], data['Wind Direction'].iloc[-1]]
            will_rain, predicted_precipitation = predict_weather(features)
            recommendation = generate_recommendation(session['soil_type'], session['crop_type'], will_rain, predicted_precipitation)
            response = recommendation
            # Reset soil and crop types for the next interaction
            session['soil_type'] = None
            session['crop_type'] = None
    
    else:
        response = "I'm sorry, I didn't understand that. Can you please ask about the weather or irrigation?"

    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)