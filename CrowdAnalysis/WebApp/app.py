from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename

import sys
sys.path.append('CrowdAnalysis/ZonalPersonCounter/')
from CrowdZonesCount import CountinZone
app = Flask(__name__)

UPLOAD_FOLDER = 'Data/Crowd_Count/ZoneCounter_Dynamic/uploads'
ALLOWED_EXTENSIONS = {'avi', 'mp4', 'mov', 'wmv', 'flv', 'mkv', 'webm', 'mpeg', '3gp', 'ts', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('zonalcrowdcount.html')


@app.route('/upload', methods=['POST'])
def handle_upload():
    new_filename = request.form['new_filename'].strip()

    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        converted_filename = new_filename + '.mp4' if new_filename else 'converted.mp4'
        converted_file_path = os.path.join(app.config['UPLOAD_FOLDER'], converted_filename)
        file.save(converted_file_path)

        # Delete the original uploaded file
        original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(original_file_path):
            os.remove(original_file_path)

        # Now process the uploaded and converted video
        CountinZone(source='file', filepath=converted_file_path)
        if os.path.exists(converted_file_path):
            os.remove(converted_file_path)
        return f'Successful!...'

    return 'Invalid file'


    return 'Invalid file'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
