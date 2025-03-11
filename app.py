from flask import Flask, render_template, request, Response, jsonify, send_file,url_for
import cv2
import os
from twilio.rest import Client
import base64
import pickle
import requests
import re
from Detectors import AttendanceManager,PersonTracker
import warnings
warnings.filterwarnings(action="ignore")
import threading
from werkzeug.utils import secure_filename
import sys
import subprocess
from email_config import EmailService,start_scheduler_thread
import time
import threading


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
attendance_manager = AttendanceManager()
RTSP_URL = 'test7.mp4' # change this to your RTSP URL
REPORTS_FOLDER = 'attendance_logs/excel_reports'



detector = PersonTracker()
current_tracks = {}  # Store tracking info
authorized_people = {} 
email_service = EmailService()

#SMS sending requiremernts
account_sid = 'ACed1d4133948a5a4c168cdef84681a025'
auth_token = 'e76fdde907597cd0878a56b76cedf7c3'
twilio_number = '+14127544576'
recipient_number = '+919737571122'
client = Client(account_sid, auth_token)
global last_sent_count
last_sent_count = 0
frame_queue = []

def video_capture_thread():
    global frame_queue
    while True:
        cap = cv2.VideoCapture(RTSP_URL)
        if not cap.isOpened():
            print("Reconnecting to camera...")
            time.sleep(2)  # Wait before retrying
            continue  # Try to reconnect

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while cap.isOpened():
            #for _ in range(1):  # Reduce skipped frames for real-time sync
            #    cap.grab()
            ret, frame = cap.read()
            if not ret:
                print("Lost connection. Reconnecting...")
                break  # Restart connection loop

            if len(frame_queue) > 1:
                frame_queue.pop(0)  # Keep only the latest frame
            frame_queue.append(frame)

        cap.release()


video_thread = threading.Thread(target=video_capture_thread, daemon=True)
video_thread.start()

def process_video():
    global current_tracks, authorized_people
    while True:
        if frame_queue:
            frame = frame_queue[-1]  # Process the latest frame
            
            # Update frame processing to include authorized status
            for track_id, info in authorized_people.items():
                if track_id in detector.tracks:
                    detector.tracks[track_id].authorized = True
                    detector.tracks[track_id].name = info['name']

            # Process frame with updated authorization status
            processed_frame = detector.process_frame(frame, 0)
            
            # Get updated track info
            current_tracks = detector.get_current_tracks()
            
            # Ensure authorized people stay authorized in current_tracks
            for track_id, info in authorized_people.items():
                if track_id in current_tracks:
                    current_tracks[track_id]['authorized'] = True
                    current_tracks[track_id]['name'] = info['name']
            
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
@app.route('/get_counts', methods=['GET'])
def get_counts():
    global current_tracks
    current_tracks = detector.get_current_tracks()

    # 🔥 Only count people who are actively being tracked
    current_people_count = len(current_tracks)

    # 🔥 Count unauthorized people separately
    unauthorized_count = sum(1 for person in current_tracks.values() if not person['authorized'])
    alert_unauthorized_person(unauthorized_count)

    return jsonify({
        'total_people': current_people_count,  # ✅ Only currently active people
        'unauthorized': unauthorized_count,    # ✅ Only currently unauthorized
        'tracks': current_tracks
    })

@app.route('/manual_override', methods=['POST'])
def manual_override():
    global authorized_people
    data = request.json
    track_id = data.get('track_id')
    name = data.get('name')
    reason = data.get('reason', '')

    try:
        # Process the override in attendance manager
        message = attendance_manager.manual_override(
            employee_id=str(track_id),
            name=name,
            reason=reason
        )
        
        # Update detector's authorized persons list
        detector.update_authorization(track_id, name)
        
        # Store authorized person in our global dictionary
        authorized_people[track_id] = {
            'name': name,
            'authorized': True
        }
        
        # Update current_tracks
        if track_id in current_tracks:
            current_tracks[track_id]['authorized'] = True
            current_tracks[track_id]['name'] = name

        return jsonify({'success': True, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400


@app.route('/', methods=['GET'])
def index():
    return render_template('index2.html')



@app.route('/video_feed')
def video_feed():
    return Response(process_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/daily_report')
def daily_report():
    report_path = attendance_manager.export_daily_report_to_excel()
    if "Error" in report_path:
        return report_path, 400
    return send_file(report_path, as_attachment=False)

@app.route('/monthly_report/<int:year>/<int:month>')
def monthly_report(year, month):
    report_path = attendance_manager.export_monthly_report_to_excel(year, month)
    if "Error" in report_path:
        return report_path, 400
    return send_file(report_path, as_attachment=False)



ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
FACES_FOLDER = 'faces'
os.makedirs(FACES_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def train_model_async():
    # First convert the dataset
    print("Model training started")
    subprocess.run([sys.executable, 'convert.py'])
    # Then train the model
    subprocess.run([sys.executable, 'train.py'])
    # Reload the detector with new model
    detector.load_model('vit_full_model.pth')



def capture_video():
    cap = cv2.VideoCapture(RTSP_URL)  # Use RTSP stream instead of webcam

    if not cap.isOpened():
        print("Error: Could not open CCTV stream.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Unable to fetch frame from CCTV.")
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()


@app.route('/capture_feed')
def capture_feed():
    return Response(capture_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register_person', methods=['POST'])
def register_person():
    try:
        data = request.json
        name = data.get('name')
        images = data.get('images', [])
        
        if not name or not images:
            return jsonify({'success': False, 'message': 'Name and images are required'}), 400
        
        person_folder = os.path.join(FACES_FOLDER, name)
        os.makedirs(person_folder, exist_ok=True)
        
        for i, img_data in enumerate(images):
            img_path = os.path.join(person_folder, f'{i}.jpg')
            with open(img_path, 'wb') as f:
                f.write(base64.b64decode(img_data.split(',')[1]))
        
        try:
            with open('auth_data.pkl', 'rb') as f:
                auth = pickle.load(f)
        except FileNotFoundError:
            auth = {}
        
        auth[name] = "Authorised"
        with open('auth_data.pkl', 'wb') as f:
            pickle.dump(auth, f)
        
        # Start model training in a separate thread
        threading.Thread(target=train_model_async, daemon=True).start()
        
        return jsonify({'success': True, 'message': f'{name} registered successfully. Model training started.'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
@app.route('/get_users', methods=['GET'])
def get_users():
    try:
        with open('auth_data.pkl', 'rb') as f:
            auth = pickle.load(f)
        return jsonify({'success': True, 'users': list(auth.keys())})
    except FileNotFoundError:
        return jsonify({'success': False, 'message': 'No users found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/update_authorization', methods=['POST'])
def update_authorization():
    try:
        data = request.json
        name = data.get('name')
        status = data.get('status')
        
        try:
            with open('auth_data.pkl', 'rb') as f:
                auth = pickle.load(f)
        except FileNotFoundError:
            auth = {}
        
        if name in auth:
            auth[name] = status
            with open('auth_data.pkl', 'wb') as f:
                pickle.dump(auth, f)
            return jsonify({'success': True, 'message': f'{name} authorization updated to {status}'})
        else:
            return jsonify({'success': False, 'message': f'User {name} not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500
    
def get_available_reports():
    daily_reports = []
    monthly_reports = []
    print("Loading reports")
    
    for f in os.listdir(REPORTS_FOLDER):
        if re.match(r"attendance_report_\d{4}-\d{2}-\d{2}\.xlsx", f):  # Matches YYYY-MM-DD format
            daily_reports.append(f)
        elif re.match(r"attendance_report_\d{4}_\d{2}\.xlsx", f):  # Matches YYYY_MM format
            monthly_reports.append(f)
    
    return {'daily': sorted(daily_reports), 'monthly': sorted(monthly_reports)}
@app.route('/get_reports', methods=['GET'])
def get_reports():
    return jsonify(get_available_reports())

@app.route('/download_report/<filename>', methods=['GET'])
def download_report(filename):
    file_path = os.path.join(REPORTS_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404
def load_email_config():
    try:
        with open('email_config.pkl', 'rb') as f:
            config = pickle.load(f)  # ✅ Ensure config is a dictionary

        # ✅ Debugging: Print loaded configuration
        print("🔄 Loaded email configuration:", config)

        if not isinstance(config, dict):  # 🚨 Ensure config is a dictionary
            raise ValueError("Invalid email configuration format: Expected a dictionary.")

        email_service.configure(
            smtp_server=config.get('smtp_server'),
            smtp_port=config.get('smtp_port'),
            sender_email=config.get('sender_email'),
            sender_password=config.get('sender_password'),  
            recipient_emails=config.get('recipient_emails', [])
        )

        print("✅ Email configuration loaded successfully!")

    except FileNotFoundError:
        print("⚠️ No email configuration file found, using defaults.")
    except Exception as e:
        print(f"❌ Error loading email configuration: {e}")  # 🔥 Debugging


@app.route('/test_email', methods=['POST'])
def test_email():
    try:
        data = request.json  # Ensure JSON is received
        if not data:
            return jsonify({"success": False, "message": "No data received"}), 400

        # Simulating email test logic
        success = True  # Replace with actual email sending logic

        if success:
            return jsonify({"success": True, "message": "Test email sent successfully!"})
        else:
            return jsonify({"success": False, "message": "Failed to send test email"}), 500

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    

@app.route('/save_email_config', methods=['POST'])
def save_email_config():
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "message": "No data received"}), 400

        # Store email configuration, including password
        config = {
            "smtp_server": data.get("smtp_server"),
            "smtp_port": data.get("smtp_port"),
            "sender_email": data.get("sender_email"),
            "sender_password": data.get("sender_password"),  # ✅ Store password
            "recipient_emails": data.get("recipient_emails", [])
        }

        # Save to pickle file
        with open('email_config.pkl', 'wb') as f:
            pickle.dump(config, f)

        # Update the live email_service instance
        email_service.configure(
            smtp_server=config["smtp_server"],
            smtp_port=config["smtp_port"],
            sender_email=config["sender_email"],
            sender_password=config["sender_password"],  # ✅ Apply stored password
            recipient_emails=config["recipient_emails"]
        )

        print("✅ Email configuration (including password) saved successfully!")
        return jsonify({"success": True, "message": "Configuration saved successfully!"})

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/send_sms', methods=['POST'])
def send_sms():
    try:
        data = request.json
        to_number = data.get('to_number', recipient_number)
        message_body = data.get('message')

        print(f"Received SMS request: To={to_number}, Message={message_body}")
        
        if not to_number or not message_body:
            return jsonify({'success': False, 'message': 'Phone number and message are required'}), 400

        message = client.messages.create(
            body=message_body,
            from_=twilio_number,
            to=to_number
        )

        print(f"SMS Sent! SID: {message.sid}")
        return jsonify({'success': True, 'message': f'Message sent with SID: {message.sid}'})
    except Exception as e:
        print(f"SMS Error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# Function to send an alert when an unauthorized person is detected
def alert_unauthorized_person(count):
    global last_sent_count  # Ensure last_sent_count is recognized globally
    if count > 0 and count > last_sent_count:
        message_body = f'ALERT: {count} unauthorized person(s) detected!'
        print(f"Triggering SMS Alert: {message_body}")
        sms_url = url_for('send_sms', _external=True)
        response = requests.post(sms_url, json={'to_number': recipient_number, 'message': message_body})
        print(f"SMS API Response: {response.status_code}, {response.text}")
        last_sent_count = count  # Update last sent count

if __name__ == '__main__':
    # Load email configuration
    load_email_config()
    
    # Start scheduler thread for automated reports and alerts
    scheduler_thread = start_scheduler_thread(
        email_service, 
        attendance_manager, 
        detector, 
        current_tracks
    )
    
    app.run(debug=True)