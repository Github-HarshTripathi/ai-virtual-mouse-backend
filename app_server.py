import os
import time
import logging
from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_virtual_mouse_hub")

# Flask app init
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# SocketIO setup
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# State tracking
STATE = {
    "clients": 0,
    "last_gesture": None,
    "last_voice": None,
    "started_at": time.time()
}

@app.route("/")
def home():
    return jsonify({
        "message": "AI Virtual Mouse Hub running",
        "uptime": int(time.time() - STATE["started_at"])
    })

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "clients": STATE["clients"],
        "last_gesture": STATE["last_gesture"],
        "last_voice": STATE["last_voice"]
    })

@socketio.on("connect")
def connect():
    STATE["clients"] += 1
    logger.info(f"Client connected ({STATE['clients']})")
    socketio.emit("connection_status", {"status": "connected", "clients": STATE["clients"]})

@socketio.on("disconnect")
def disconnect():
    STATE["clients"] = max(0, STATE["clients"] - 1)
    logger.info(f"Client disconnected ({STATE['clients']})")

@socketio.on("gesture_event")
def gesture_event(data):
    STATE["last_gesture"] = data
    logger.info(f"Gesture: {data}")
    socketio.emit("gesture_broadcast", data, broadcast=True)

@socketio.on("voice_event")
def voice_event(data):
    STATE["last_voice"] = data
    logger.info(f"Voice: {data}")
    socketio.emit("voice_broadcast", data, broadcast=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)
