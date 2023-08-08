import os
import cv2
from flask import Flask, render_template, request, redirect, url_for, session
import sys
sys.path.append('Facial_Recognition/')
from Database_Connect.Load_Users import insert_user_info
def generate_secret_key(length=32):
    return os.urandom(length).hex()


app = Flask(__name__)
app.secret_key = generate_secret_key()

# Folder to store recorded videos
UPLOAD_FOLDER = 'Data/Facial_Recog/Raw_DataStore/FacialRecog_TargetVideo'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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

        session['FirstName'] = first_name
        session['LastName'] = last_name

        user_info = {
            'FirstName': first_name,
            'LastName': last_name,
            'Age': age,
            'Ethnicity': gender,
            'Gender': ethnicity
        }

        insert_user_info(user_info)

        return redirect(url_for('record_video_page'))

    return render_template('record_mode.html')

@app.route('/record_video_page', methods=['GET', 'POST'])
def record_video_page():
    return render_template('record_video_page.html')

@app.route('/record_video', methods=['GET', 'POST'])
def record_video():
    first_name = session.get('FirstName', '')
    last_name = session.get('LastName', '')
    if request.method == 'POST':
        video_path = os.path.join(UPLOAD_FOLDER, f'{first_name}_{last_name}.mp4')
        capture = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_width = int(capture.get(3))
        frame_height = int(capture.get(4))
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (frame_width, frame_height))

        # Record for 10 seconds
        start_time = cv2.getTickCount()
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break

            out.write(frame)
            cv2.imshow('Recording', frame)

            # Stop recording after 10 seconds
            if (cv2.getTickCount() - start_time) / cv2.getTickFrequency() > 10:
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        capture.release()
        out.release()
        cv2.destroyAllWindows()

        return "Video recorded successfully"
    return render_template('record_video.html')

@app.route('/test_mode')
def test_mode():
    return "Test Mode Page"


if __name__ == '__main__':
    app.run(debug=True)

        # # Capture video
        # video_path = os.path.join(UPLOAD_FOLDER, f'{first_name}_{last_name}.mp4')
        # capture = cv2.VideoCapture(0)
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
        #
        # # Record for 10 seconds
        # while capture.isOpened():
        #     ret, frame = capture.read()
        #     if not ret:
        #         break
        #
        #     out.write(frame)
        #     cv2.imshow('Recording', frame)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        #
        # capture.release()
        # out.release()
        # cv2.destroyAllWindows()
        #
        # return f'Recorded and `saved video as {video_path}<br><pre>{user_info}</pre>'`



if __name__ == '__main__':
    app.run(debug=True)
