from flask import Flask, render_template, Response, request, redirect, url_for, session
import cv2
from ultralytics import YOLO
import time
import winsound
import requests

app = Flask(__name__)
app.secret_key = "railway_ai_secret"   # 🔐 session key

# ---------------- LOGIN CREDENTIALS ----------------
USERNAME = "stationmaster"
PASSWORD = "railway123"

# 📩 FAST2SMS SETUP
FAST2SMS_API_KEY = "YOUR_API_KEY"
STATION_MASTER_NUMBER = "9061642481,9072214249"

def send_sms_fast2sms(object_name):
    url = "https://www.fast2sms.com/dev/bulk"
    payload = {
        "sender_id": "TXTIND",
        "message": f"🚨 ALERT! Obstacle detected: {object_name}",
        "language": "english",
        "route": "p",
        "numbers": STATION_MASTER_NUMBER
    }
    headers = {
        "authorization": FAST2SMS_API_KEY,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    try:
        requests.post(url, data=payload, headers=headers, timeout=5)
    except:
        pass

# 🔔 GLOBAL STATUS
alert_status = "SAFE"
last_detected_object = "None"
sms_sent = False

# Load YOLO model
model = YOLO("yolov8n.pt")

# Camera
camera_url = "http://10.143.96.43:8080/video"
cap = cv2.VideoCapture(camera_url)

danger_objects = ["cow", "dog", "horse", "elephant", "person", "car", "truck", "bus"]
last_beep_time = 0

def generate_frames():
    global last_beep_time, alert_status, last_detected_object, sms_sent

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame, conf=0.5)
        detected = False
        label = ""

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                if label in danger_objects:
                    detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"WARNING: {label.upper()}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)

        if detected:
            alert_status = "DANGER"
            last_detected_object = label

            if not sms_sent:
                send_sms_fast2sms(label)
                sms_sent = True

            if time.time() - last_beep_time > 2:
                winsound.Beep(1000, 300)
                last_beep_time = time.time()
        else:
            alert_status = "SAFE"
            last_detected_object = "None"
            sms_sent = False

        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")

# ---------------- ROUTES ----------------

@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form["username"]
        pwd = request.form["password"]

        if user == USERNAME and pwd == PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("station"))
        else:
            return render_template("login.html", error="Invalid Login")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))

@app.route("/station")
def station():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("station_master.html")

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    return {
        "status": alert_status,
        "object": last_detected_object,
        "time": time.strftime("%H:%M:%S")
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)