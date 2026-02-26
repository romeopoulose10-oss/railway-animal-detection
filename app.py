import os
import cv2
import time
import requests
from flask import Flask, render_template, Response, request, redirect, url_for, session
from ultralytics import YOLO

# ---------------- ENV CHECK ----------------
RENDER = os.getenv("RENDER") is not None

app = Flask(__name__)
app.secret_key = "railway_ai_secret"

# ---------------- LOGIN ----------------
USERNAME = "stationmaster"
PASSWORD = "railway123"

# ---------------- SMS CONFIG ----------------
FAST2SMS_API_KEY = "YOUR_API_KEY"
STATION_MASTER_NUMBER = "9061642481,9072214249"

def send_sms_fast2sms(object_name):
    if RENDER:
        return  # ❌ disable SMS in cloud demo
    try:
        requests.post(
            "https://www.fast2sms.com/dev/bulk",
            data={
                "sender_id": "TXTIND",
                "message": f"🚨 ALERT! Obstacle detected: {object_name}",
                "language": "english",
                "route": "p",
                "numbers": STATION_MASTER_NUMBER
            },
            headers={"authorization": FAST2SMS_API_KEY},
            timeout=5
        )
    except:
        pass

# ---------------- GLOBAL STATE ----------------
alert_status = "SAFE"
last_detected_object = "None"
sms_sent = False

# ---------------- LOAD MODEL (LOCAL ONLY) ----------------
model = None
cap = None

if not RENDER:
    model = YOLO("yolov8n.pt")
    camera_url = "http://10.159.192.219:8080/video"
    cap = cv2.VideoCapture(camera_url)
    print("📷 Camera enabled (Local)")
else:
    print("☁️ Running on Render → Camera disabled")

danger_objects = ["cow", "dog", "horse", "elephant", "person", "car", "truck", "bus"]

# ---------------- VIDEO STREAM ----------------
def generate_frames():
    global alert_status, last_detected_object, sms_sent

    if RENDER or cap is None:
        while True:
            time.sleep(1)
            yield b""
        return

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
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if detected:
            alert_status = "DANGER"
            last_detected_object = label
            if not sms_sent:
                send_sms_fast2sms(label)
                sms_sent = True
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
        if request.form["username"] == USERNAME and request.form["password"] == PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("station"))
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

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

