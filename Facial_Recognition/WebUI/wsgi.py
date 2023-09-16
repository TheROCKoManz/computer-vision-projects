import sys
sys.path.append('Facial_Recognition/WebUI')


from app import app, socketio

if __name__ == '__main__':
    socketio.run(app, debug=True)
