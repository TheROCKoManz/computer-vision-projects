import os
import cv2
from flask import Flask, render_template, request
import sys
sys.path.append('Facial_Recognition/')
from Database_Connect.Load_Users import insert_user_info

app = Flask(__name__)

# Folder to store recorded videos
UPLOAD_FOLDER = 'Data/Facial_Recog/Raw_DataStore/FacialRecog_TargetVideo'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def record_mode():
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form['last_name']
        age = request.form['age']
        gender = request.form['gender']
        ethnicity = request.form['ethnicity']

        user_info = {
            'first_name': first_name,
            'last_name': last_name,
            'age': age,
            'gender': gender,
            'ethnicity': ethnicity
        }

        insert_user_info(user_info)



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
        # return f'Recorded and saved video as {video_path}<br><pre>{user_info}</pre>'

    return render_template('record_mode.html')

if __name__ == '__main__':
    app.run(debug=True)
