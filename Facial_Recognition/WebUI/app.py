import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, session
import sys
import base64
from flask_sslify import SSLify
from flask_socketio import SocketIO, emit
import numpy as np
sys.path.append('Facial_Recognition/')
from Database_Connect.Load_Users import insert_user_info, insert_user_encodings
from DataPreproc.extractFaceEncodings import get_embedding_train
from Server_Loading.User_Video_counts import count_user_videos
from Server_Loading.Upload_solo_file_to_Server import upload_filex
from Prediction import fetch_encodings_from_db, Frame_Face_Recognition

def generate_secret_key(length=32):
    return os.urandom(length).hex()


app = Flask(__name__)
app.secret_key = generate_secret_key()
# sslify = SSLify(app)
socketio = SocketIO(app)
# Folder to store recorded videos
UPLOAD_FOLDER = 'Data/Facial_Recog/Raw_DataStore/FacialRecog_TargetVideo'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
user_ids, first_names, last_names, face_encodings = fetch_encodings_from_db()

@app.route('/')
def main_page():
    return render_template('index.html')


@app.route('/record_mode', methods=['GET', 'POST'])
def record_mode():
    if request.method == 'POST':
        first_name = request.form['FirstName']
        last_name = request.form['LastName']
        age = request.form['Age']
        gender = request.form['Ethnicity']
        ethnicity = request.form['Gender']

        user_info = {
            'FirstName': first_name,
            'LastName': last_name,
            'Age': age,
            'Ethnicity': gender,
            'Gender': ethnicity
        }

        scenario,userID = insert_user_info(user_info)

        if scenario==1:
            return redirect(url_for('user_exists_continue_to_record', userID = userID))

        else:
            return redirect(url_for('user_inserted_continue_to_record', userID = userID))

    return render_template('record_mode.html')

@app.route('/user_inserted_continue_to_record', methods=['GET', 'POST'])
def user_inserted_continue_to_record():
    userID = request.args.get('userID', '')
    return render_template('user_inserted_continue_to_record.html', userID = userID)

@app.route('/user_exists_continue_to_record', methods=['GET', 'POST'])
def user_exists_continue_to_record():
    userID = request.args.get('userID', '')
    return render_template('user_exists_continue_to_record.html', userID=userID)


@app.route('/record_video', methods=['GET', 'POST'])
def record_video():
    userID = request.args.get('userID', '')
    print('UserID for encoding: ', userID)
    video_count = count_user_videos(userID)
    video_filename = f'{userID}_{str(video_count + 1)}'

    if request.method == 'POST':
        video_data_base64 = request.form['videoData']

        # Add padding characters if needed
        while len(video_data_base64) % 4 != 0:
            video_data_base64 += '='

        video_data = base64.b64decode(video_data_base64)

        # Save the video data as a webm file
        webm_filename = f'{video_filename}.webm'
        webm_path = os.path.join(UPLOAD_FOLDER, webm_filename)
        with open(webm_path, 'wb') as f:
            f.write(video_data)

        # Convert the webm video to mp4
        mp4_filename = f'{video_filename}.mp4'
        mp4_path = os.path.join(UPLOAD_FOLDER, mp4_filename)
        command = f'ffmpeg -i {webm_path} {mp4_path}'
        os.system(command)
        upload_filex(mp4_path)
        face_encoding = get_embedding_train([userID])
        insert_user_encodings(userID, face_encoding)
        os.remove(webm_path)
        return render_template('CompletedTraining.html')

    return render_template('record_video.html')


def base64_to_image(base64_string):
    # Extract the base64 encoded binary data from the input string
    base64_data = base64_string.split(",")[1]
    # Decode the base64 data to bytes
    image_bytes = base64.b64decode(base64_data)
    # Convert the bytes to numpy array
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    # Decode the numpy array as an image using OpenCV
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image

# Socket.IO event
@socketio.on('connect')
def connect():
    print('Client connected')

@socketio.on('disconnect')
def disconnect():
    print('Client disconnected')


# Route for the prediction page
@app.route('/predict')
def predict():
    return render_template('predict.html')


# Socket.IO event for processing frames and sending processed frames back to the client
@socketio.on('image')
def handle_frame(image_data):
    try:
        # Decode the base64 image data
        image = base64_to_image(image_data)
        pred_frame = Frame_Face_Recognition(image, user_ids, first_names, last_names, face_encodings)
        pred_frame = cv2.resize(pred_frame, (600,400))
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, frame_encoded = cv2.imencode(".jpg", pred_frame, encode_param)
        processed_img_data = base64.b64encode(frame_encoded).decode()
        b64_src = "data:image/jpg;base64,"
        processed_img_data = b64_src + processed_img_data
        emit("processed_image", processed_img_data)
    except Exception as e:
        print("Error handling frame:", e)


@app.route('/restart', methods=['GET'])
def restart():
    return redirect(url_for('main_page'))


if __name__ == '__main__':
    # app.run(host='0.0.0.0', debug=True, port=0, ssl_context=('localhost.crt', 'localhost.key'))
    socketio.run(app, host='0.0.0.0', debug=True, port=4269)
