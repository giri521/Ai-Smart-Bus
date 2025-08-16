# app.py
# Import necessary modules
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import io
import os
from PIL import Image
from twilio.rest import Client
import re # Import the regular expressions module

# Initialize the Flask application
app = Flask(__name__)

# --- Translations for the ULTRA-CONCISE SMS message ---
SMS_TRANSLATIONS = {
    'en-IN': "Bus {bus_number} | Route: {route} | Seats: {seats} | Wx: {weather} | {link}",
    'hi-IN': "बस {bus_number} | मार्ग: {route} | सीटें: {seats} | मौसम: {weather} | {link}",
    'te-IN': "బస్సు {bus_number} | మార్గం: {route} | సీట్లు: {seats} | వాతా: {weather} | {link}",
    'ta-IN': "பஸ் {bus_number} | வழி: {route} | இடங்கள்: {seats} | வானிலை: {weather} | {link}"
}

# --- AI Model Loading ---
# Load the pre-trained YOLOv8n model
try:
    model = YOLO('yolov8n.pt')
    print("YOLOv8n model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

twilio_client = None

if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER:
    try:
        from twilio.rest import Client
        twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        print("Twilio client initialized successfully.")
    except Exception as e:
        print(f"Error initializing Twilio client: {e}")
        twilio_client = None
else:
    print("Twilio credentials not provided. SMS functionality will be disabled.")

# --- HTML Page Routes ---
@app.route('/')
def index(): return render_template('index.html')
@app.route('/user')
def user(): return render_template('user.html')
@app.route('/bus')
def bus(): return render_template('bus.html')
@app.route('/add_bus')
def add_bus(): return render_template('add_bus.html')
@app.route('/driver_dashboard')
def driver_dashboard(): return render_template('driver_dashboard.html')


# --- API Endpoints ---
@app.route('/detect_crowd', methods=['POST'])
def detect_crowd():
    if not model: return jsonify({'error': 'AI Model is not available on the server'}), 500
    data = request.get_json()
    if not data or 'image' not in data: return jsonify({'error': 'No image data provided'}), 400
    try:
        _, image_b64_string = data['image'].split(',', 1)
        image = Image.open(io.BytesIO(base64.b64decode(image_b64_string)))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except Exception as e: return jsonify({'error': f'Failed to process image: {e}'}), 400
    results = model(frame)
    crowd_count = sum(1 for r in results for box in r.boxes if model.names[int(box.cls)] == 'person')
    return jsonify({'crowd_count': crowd_count})

@app.route('/send_sms', methods=['POST'])
def send_sms():
    if not twilio_client:
        return jsonify({'success': False, 'error': 'Twilio service is not configured on the server.'}), 500

    data = request.get_json()
    phone_number = data.get('phone_number')
    bus_details = data.get('bus_details')
    language = data.get('language', 'en-IN') # Get language, default to English

    if not phone_number or not bus_details:
        return jsonify({'success': False, 'error': 'Missing phone number or bus details.'}), 400

    try:
        # Sanitize weather data to remove Unicode characters
        weather_str = bus_details.get('weather', 'N/A')
        match = re.search(r'[\d.]+', weather_str)
        sanitized_weather = match.group(0) + 'c' if match else 'N/A'

        # --- CORRECTED LIVE LOCATION LINK ---
        lat = bus_details.get('lat')
        lon = bus_details.get('lon')
        maps_link = f"https://maps.google.com/?q={lat},{lon}" if lat and lon else "Not available"

        # Get the correct message template based on language
        message_template = SMS_TRANSLATIONS.get(language, SMS_TRANSLATIONS['en-IN'])
        
        # Format the message using the chosen template
        message_body = message_template.format(
            bus_number=bus_details.get('busNumber', 'N/A'),
            route=bus_details.get('route', 'N/A'),
            seats=bus_details.get('availableSeats', 'N/A'),
            weather=sanitized_weather,
            link=maps_link
        )

        message = twilio_client.messages.create(
            body=message_body,
            from_=TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        
        print(f"SMS sent successfully to {phone_number} in {language}, SID: {message.sid}")
        return jsonify({'success': True, 'message': 'SMS sent successfully!'})

    except Exception as e:
        print(f"Error sending SMS: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':

    app.run(debug=True)




