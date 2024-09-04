
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import torch
import numpy as np
import secrets
import torch.nn as nn
import os
from twilio.rest import Client  # Import Twilio Client

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = secrets.token_hex(16)
socketio = SocketIO(app)

# Twilio credentials
TWILIO_ACCOUNT_SID = 'put your own credential'
TWILIO_AUTH_TOKEN = 'put your own credential'
TWILIO_PHONE_NUMBER = 'put your own credential'
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

class MaintenanceModel(nn.Module):
    def __init__(self):
        super(MaintenanceModel, self).__init__()
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def simulate_realtime_data():
    while True:
        num_rows = np.random.randint(1, 10)
        data = np.random.rand(num_rows, 6)
        yield data

# Initialize alert_sent flag
alert_sent = False

def predict_and_alert(model):
    global alert_sent  # Use global variable
    for data in simulate_realtime_data():
        with torch.no_grad():
            inputs = torch.tensor(data, dtype=torch.float32)
            predictions = model(inputs)
            binary_predictions = (predictions > 0.5).float()
            prediction_values = binary_predictions.squeeze().tolist()

            if isinstance(prediction_values, float):
                prediction_values = [prediction_values]

            for pred in prediction_values:
                if pred == 1 and not alert_sent:  # Check if alert hasn't been sent
                    emit('alert_message', {'message': 'Failure predicted!'})
                    send_sms_alert()
                    alert_sent = True  # Update flag to indicate alert has been sent

        socketio.sleep(1)

def send_sms_alert():
    try:
        message = client.messages.create(
            body='Failure predicted! Please take necessary action. click to check the solution : {message}  link-----------------',
            from_=TWILIO_PHONE_NUMBER,
            to='***********'  # Replace with actual customer phone number
        )
        print(f'SMS alert sent successfully: {message.sid}')
    except Exception as e:
        print(f'Error sending SMS alert: {e}')

@app.route('/')
def index():
    print(os.getcwd())
    print(os.listdir('templates'))
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    try:
        model_state_dict = torch.load('predictive_maintenance_model.pt', map_location=torch.device('cpu'))
        model = MaintenanceModel()
        model.load_state_dict(model_state_dict)
        model.eval()
        predict_and_alert(model)
    except Exception as e:
        app.logger.error(f'Error loading model: {e}')
        emit('alert_message', {'message': 'Error loading model!'})

if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
