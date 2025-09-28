# Save as: backend/verify_installation.py
"""
Installation verification script for AI Virtual Mouse
Run this to check if all dependencies are properly installed
"""

import sys
import importlib.util

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} - OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name} - ERROR: {e}")
        return False

def main():
    print("🔍 Verifying AI Virtual Mouse Backend Dependencies...")
    print("=" * 50)
    
    packages = [
        ('Flask', 'flask'),
        ('Flask-CORS', 'flask_cors'),
        ('Flask-SocketIO', 'flask_socketio'),
        ('OpenCV', 'cv2'),
        ('MediaPipe', 'mediapipe'),
        ('PyAutoGUI', 'pyautogui'),
        ('NumPy', 'numpy'),
        ('SpeechRecognition', 'speech_recognition'),
        ('pyttsx3', 'pyttsx3'),
        ('Pillow', 'PIL'),
        ('PyAudio', 'pyaudio'),
        ('Eventlet', 'eventlet'),
        ('Requests', 'requests')
    ]
    
    success_count = 0
    total_count = len(packages)
    
    for package_name, import_name in packages:
        if check_package(package_name, import_name):
            success_count += 1
    
    print("=" * 50)
    print(f"📊 Results: {success_count}/{total_count} packages successfully installed")
    
    if success_count == total_count:
        print("🎉 All dependencies are properly installed!")
        print("✨ Your AI Virtual Mouse backend is ready to run!")
        
        # Test camera availability
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("📷 Camera detected and accessible")
                cap.release()
            else:
                print("⚠️  Camera not detected - please check camera connection")
        except Exception as e:
            print(f"⚠️  Camera test failed: {e}")
            
    else:
        print("⚠️  Some dependencies are missing. Please run:")
        print("   pip install -r requirements.txt")
    
    # Python version check
    python_version = sys.version_info
    print(f"\n🐍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major >= 3 and python_version.minor >= 8:
        print("✅ Python version is compatible")
    else:
        print("⚠️  Python 3.8+ recommended for best compatibility")

if __name__ == "__main__":
    main()
