from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from ai_virtual_mouse import ai_mouse
import cv2
import threading
import time
import logging
import os
import sys
import numpy as np
import signal


# -----------------------
# Logging configuration
# -----------------------
LOG_FILE = 'ai_mouse.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('ai_virtual_mouse_backend')


# -----------------------
# Flask + SocketIO setup
# -----------------------
app = Flask(__name__)

# Example: replace '*' with your frontend domain for security
CORS(app, resources={r"/*": {"origins": ["https://yourfrontend.vercel.app"]}})

socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', logger=True, engineio_logger=True)


# -----------------------
# Thread-safe application state
# -----------------------
class AppState:
    def __init__(self):
        self.is_gesture_running = False
        self.is_voice_running = False
        self.gesture_thread = None
        self.voice_thread = None
        self.lock = threading.Lock()


app_state = AppState()


# -----------------------
# Helpers
# -----------------------
def safe_float(val, default):
    try:
        return float(val)
    except Exception:
        return default


def safe_int(val, default):
    try:
        return int(val)
    except Exception:
        return default


def create_error_frame(message, width=640, height=480):
    """Create a visually clear error frame (numpy array)"""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        color = int(30 + (i / height) * 70)
        cv2.line(frame, (0, i), (width, i), (color, color, color), 1)
    cv2.putText(frame, "AI Virtual Mouse", (max(20, width//2 - 180), max(40, height//2 - 60)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, message, (max(20, width//2 - 220), height//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Check camera connection & permissions", (max(20, width//2 - 260), height//2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


def add_ui_overlay(frame, gesture):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    rect_h = 110
    cv2.rectangle(overlay, (10, 10), (w - 10, rect_h), (0, 0, 0), -1)
    alpha = 0.35
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, "AI Virtual Mouse - Active", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Gesture: {gesture}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {'Active' if app_state.is_gesture_running else 'Inactive'}", (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def safe_join_thread(thread, timeout=2.0):
    if thread and thread.is_alive():
        thread.join(timeout)


# -----------------------
# Routes
# -----------------------
@app.route('/')
def home():
    return jsonify({
        "message": "AI Virtual Mouse Backend is running!",
        "status": "active",
        "version": "2.3.0",
        "endpoints": {
            "start_gesture": "/start_gesture (POST)",
            "stop_gesture": "/stop_gesture (POST)",
            "start_voice": "/start_voice (POST)",
            "stop_voice": "/stop_voice (POST)",
            "video_feed": "/video_feed (GET)",
            "status": "/status (GET)",
            "health": "/health (GET)"
        }
    })


@app.route('/health')
def health():
    try:
        camera_available = False
        camera_status = "not_initialized"
        if getattr(ai_mouse, 'cap', None):
            camera_available = bool(ai_mouse.cap.isOpened())
            camera_status = "connected" if camera_available else "disconnected"

        return jsonify({
            "status": "healthy",
            "camera_available": camera_available,
            "camera_status": camera_status,
            "gesture_running": app_state.is_gesture_running,
            "voice_running": app_state.is_voice_running,
            "timestamp": time.time(),
            "system_load": os.getloadavg() if hasattr(os, 'getloadavg') else None
        })
    except Exception as e:
        logger.exception("Health check failed")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/status')
def status():
    return jsonify({
        "gesture_running": app_state.is_gesture_running,
        "voice_running": app_state.is_voice_running,
        "backend_connected": True,
        "current_gesture": ai_mouse.current_gesture.value if getattr(ai_mouse, 'current_gesture', None) else "NO_HAND",
        "camera_connected": getattr(ai_mouse, 'cap', None) and ai_mouse.cap.isOpened()
    })


# --- Gesture control routes ---
@app.route('/start_gesture', methods=['POST'])
def start_gesture():
    with app_state.lock:
        if app_state.is_gesture_running:
            return jsonify({"status": "already_running", "message": "Gesture recognition already running"})

        try:
            data = request.get_json() or {}
            sensitivity = safe_float(data.get('sensitivity', 1.2), 1.2)
            smoothing = safe_int(data.get('smoothing', 5), 5)

            sensitivity = max(0.1, min(3.0, sensitivity))
            smoothing = max(1, min(15, smoothing))

            ai_mouse.sensitivity = sensitivity
            ai_mouse.smoothing_factor = smoothing

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if not getattr(ai_mouse, 'cap', None) or not ai_mouse.cap.isOpened():
                        ai_mouse.initialize_camera()
                    if getattr(ai_mouse, 'cap', None) and ai_mouse.cap.isOpened():
                        break
                    logger.warning("Camera not ready, retrying...")
                    time.sleep(1)
                except Exception as e:
                    logger.warning(f"Camera init attempt {attempt+1} error: {e}")
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)

            if not getattr(ai_mouse, 'cap', None) or not ai_mouse.cap.isOpened():
                logger.error("Camera unavailable after retries")
                return jsonify({"status": "error", "message": "Camera not available. Please check the connection."}), 500

            app_state.gesture_thread = threading.Thread(
                target=ai_mouse.process_gestures,
                args=(socketio,),
                daemon=True,
                name="GestureProcessor"
            )
            app_state.gesture_thread.start()
            app_state.is_gesture_running = True

            logger.info(f"Gesture recognition started (sensitivity={sensitivity}, smoothing={smoothing})")
            return jsonify({
                "status": "started",
                "message": "Gesture recognition started",
                "sensitivity": ai_mouse.sensitivity,
                "smoothing": ai_mouse.smoothing_factor,
                "camera_resolution": f"{int(ai_mouse.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(ai_mouse.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
            })
        except Exception as e:
            logger.exception("Failed to start gesture recognition")
            app_state.is_gesture_running = False
            return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/stop_gesture', methods=['POST'])
def stop_gesture():
    with app_state.lock:
        try:
            setattr(ai_mouse, 'is_active', False)
            app_state.is_gesture_running = False
            if getattr(ai_mouse, 'is_dragging', False):
                ai_mouse.is_dragging = False
            safe_join_thread(app_state.gesture_thread, timeout=2.0)
            logger.info("Gesture recognition stopped")
            return jsonify({"status": "stopped", "message": "Gesture recognition stopped"})
        except Exception as e:
            logger.exception("Error stopping gesture recognition")
            return jsonify({"status": "error", "message": str(e)}), 500


# --- Voice control routes ---
@app.route('/start_voice', methods=['POST'])
def start_voice():
    with app_state.lock:
        if app_state.is_voice_running:
            return jsonify({"status": "already_running", "message": "Voice recognition already running"})

        try:
            if not getattr(ai_mouse, 'voice_recognizer', None):
                ai_mouse.initialize_voice()

            if not getattr(ai_mouse, 'voice_recognizer', None):
                logger.error("Voice recognizer not available after initialization")
                return jsonify({"status": "error", "message": "Voice recognition not available. Check microphone."}), 500

            app_state.voice_thread = threading.Thread(
                target=ai_mouse.start_voice_recognition,
                args=(socketio,),
                daemon=True,
                name="VoiceProcessor"
            )
            app_state.voice_thread.start()
            app_state.is_voice_running = True
            logger.info("Voice recognition started")
            return jsonify({"status": "started", "message": "Voice recognition started"})
        except Exception as e:
            logger.exception("Failed to start voice recognition")
            app_state.is_voice_running = False
            return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/stop_voice', methods=['POST'])
def stop_voice():
    with app_state.lock:
        try:
            if hasattr(ai_mouse, 'voice_mode'):
                ai_mouse.voice_mode = False
            app_state.is_voice_running = False
            safe_join_thread(app_state.voice_thread, timeout=2.0)
            logger.info("Voice recognition stopped")
            return jsonify({"status": "stopped", "message": "Voice recognition stopped"})
        except Exception as e:
            logger.exception("Error stopping voice recognition")
            return jsonify({"status": "error", "message": str(e)}), 500


# -----------------------
# Video feed route
# -----------------------
@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            try:
                if not getattr(ai_mouse, 'cap', None) or not ai_mouse.cap.isOpened():
                    error_frame = create_error_frame("Camera not available")
                    ret, jpeg = cv2.imencode('.jpg', error_frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    time.sleep(0.5)
                    continue

                ret, frame = ai_mouse.cap.read()
                if not ret or frame is None:
                    logger.debug("Failed to read frame from camera")
                    time.sleep(0.02)
                    continue

                frame = cv2.flip(frame, 1)

                if getattr(ai_mouse, 'hands', None) and getattr(ai_mouse, 'mp_draw', None) and getattr(ai_mouse, 'mp_hands', None):
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = ai_mouse.hands.process(frame_rgb)
                    if results and getattr(results, 'multi_hand_landmarks', None):
                        for hand_landmarks in results.multi_hand_landmarks:
                            ai_mouse.mp_draw.draw_landmarks(
                                frame, hand_landmarks, ai_mouse.mp_hands.HAND_CONNECTIONS,
                                ai_mouse.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                                ai_mouse.mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                            )

                gesture_text = ai_mouse.current_gesture.value if getattr(ai_mouse, 'current_gesture', None) else "NO_HAND"
                add_ui_overlay(frame, gesture_text)

                ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

                time.sleep(0.03)
            except Exception as e:
                logger.exception("Error in video feed generator")
                time.sleep(0.1)

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# -----------------------
# Socket events
# -----------------------
@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')
    socketio.emit('connection_status', {
        'status': 'connected',
        'message': 'Welcome to AI Virtual Mouse!',
        'timestamp': time.time(),
        'version': '2.3.0'
    })


@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')


@socketio.on('adjust_settings')
def handle_adjust_settings(data):
    try:
        if 'sensitivity' in data:
            ai_mouse.sensitivity = max(0.1, min(3.0, float(data['sensitivity'])))
        if 'smoothing' in data:
            ai_mouse.smoothing_factor = max(1, min(15, int(data['smoothing'])))

        socketio.emit('settings_updated', {
            'sensitivity': ai_mouse.sensitivity,
            'smoothing': ai_mouse.smoothing_factor
        })
        logger.info(f"Settings updated: sensitivity={ai_mouse.sensitivity}, smoothing={ai_mouse.smoothing_factor}")
    except Exception:
        logger.exception("Failed to adjust settings")


@socketio.on('ping')
def handle_ping():
    socketio.emit('pong', {'timestamp': time.time()})


# -----------------------
# Graceful shutdown
# -----------------------
def shutdown_handler(signum, frame):
    logger.info("Shutdown signal received, cleaning up...")
    try:
        with app_state.lock:
            if app_state.is_gesture_running:
                setattr(ai_mouse, 'is_active', False)
                app_state.is_gesture_running = False
            if app_state.is_voice_running:
                if hasattr(ai_mouse, 'voice_mode'):
                    ai_mouse.voice_mode = False
                app_state.is_voice_running = False

        safe_join_thread(app_state.gesture_thread, timeout=2.0)
        safe_join_thread(app_state.voice_thread, timeout=2.0)

        try:
            ai_mouse.cleanup()
        except Exception:
            logger.exception("ai_mouse.cleanup failed")
    except Exception:
        logger.exception("Error during shutdown")
    finally:
        logger.info("Shutdown complete")
        sys.exit(0)


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


# -----------------------
# Run server (FINAL for production)
# -----------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'

    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

    print(f"🚀 Starting AI Virtual Mouse Backend on {host}:{port}")
    print(f"🌐 Environment: {'PRODUCTION' if port != 5000 else 'DEVELOPMENT'}")

    try:
        if debug_mode or port == 5000:
            socketio.run(app, host=host, port=port, debug=debug_mode)
        else:
            try:
                import eventlet
                eventlet.monkey_patch()
                socketio.run(app, host=host, port=port, debug=False)
            except ImportError:
                print("⚠️ Eventlet not available, using development server")
                socketio.run(app, host=host, port=port, debug=False)
    except Exception as e:
        logger.exception("Server failed to start")
        try:
            ai_mouse.cleanup()
        except Exception:
            logger.exception("ai_mouse.cleanup failed on server error")
        sys.exit(1)
