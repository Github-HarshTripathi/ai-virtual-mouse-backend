# File: backend/railway_app.py - PRODUCTION READY
import os
import logging
from app import app, socketio

# Configure production settings
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = '0.0.0.0'
    
    print(f"🚀 Starting AI Virtual Mouse Backend on {host}:{port}")
    print(f"🌐 Environment: {'PRODUCTION' if port != 5000 else 'DEVELOPMENT'}")
    
    # Use eventlet for production if available
    try:
        import eventlet
        eventlet.monkey_patch()
        socketio.run(app, host=host, port=port, debug=False)
    except ImportError:
        print("⚠️ Eventlet not available, using development server")
        app.run(host=host, port=port, debug=False)