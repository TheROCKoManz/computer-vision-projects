import os
from flask import Flask, render_template, request, redirect, url_for, session
import sys
import base64
from flask_sslify import SSLify

sys.path.append('Facial_Recognition/')
from Database_Connect.Load_Users import insert_user_info
from Server_Loading.User_Video_counts import count_user_videos
from Server_Loading.Upload_solo_file_to_Server import upload_filex
from TrainingPipeline import train

def generate_secret_key(length=32):
    return os.urandom(length).hex()


app = Flask(__name__)
app.secret_key = generate_secret_key()
sslify = SSLify(app)
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
    return render_template('user_exists_continue_to_record.html', userID = userID)


@app.route('/record_video', methods=['GET', 'POST'])
def record_video():
    userID = request.args.get('userID', '')
    video_count = count_user_videos(userID)
    video_filename = f'{userID}_{str(video_count + 1)}.mp4'

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

        os.remove(webm_path)
        return "Video recorded and saved successfully"

    return render_template('record_video.html')


@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    if request.method == 'POST':
        user_ids = request.form.get('userIds')
        # Split the user IDs string and convert it to a list
        # Redirect to the model_training_progress page and pass the targets as parameters
        return redirect(url_for('train_loading', targets=user_ids))

    return render_template('train_model.html')

@app.route('/train_loading', methods=['GET'])
def train_loading():
    targets = request.args.get('targets')
    return render_template('train_loading.html', targets=targets)

@app.route('/model_training_progress', methods=['GET'])
def model_training_progress():
    targets = request.args.get('targets')  # Get the list of targets from the URL parameters
    print(targets)
    targets = [target.strip() for target in targets.strip('[]').split(',')]
    print(targets)
    # Call your training function with the extracted targets
    model_id = train(targets)
    # model_id = 'MODEL' ##Debug
    # Redirect to the training_complete page and pass the generated model_id as a parameter
    return redirect(url_for('training_complete', model_id=model_id))

@app.route('/training_complete', methods=['GET'])
def training_complete():
    model_id = request.args.get('model_id',)
    return render_template('training_complete.html', model_id=model_id)


@app.route('/restart', methods=['GET'])
def restart():
    return redirect(url_for('main_page'))

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True, port=4200, ssl_context=('localhost.crt', 'localhost.key'))