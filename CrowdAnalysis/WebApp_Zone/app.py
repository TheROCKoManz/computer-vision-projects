#### TEST VISION........###

import base64
import shutil

import sys
sys.path.append('CrowdAnalysis/WebApp_Zone/')
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, Response
import os
from werkzeug.utils import secure_filename
from CrowdZonesCount import get_frame, setup_zones, process_frame, Model
app = Flask(__name__)
from supervision import get_video_frames_generator
UPLOAD_FOLDER = 'Data/Crowd_Count/ZoneCounter_Dynamic/uploads'
ALLOWED_EXTENSIONS = {'avi', 'mp4', 'mov', 'wmv', 'flv', 'mkv', 'webm', 'mpeg', '3gp', 'ts', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

class ZoneVision:
    def __init__(self):
        self.video_path = ''
        self.all_zones = []
        self.zone_annotators = []
        self.box_annotators = []
        self.model = Model()

VisionObject = ZoneVision()


def generate_secret_key(length=32):
    return os.urandom(length).hex()


app.secret_key = generate_secret_key()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def main_page():
    return render_template('zonalcrowdcount.html')

@app.route('/camera')
def camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Scope for Camera Access"
    cap.release()
    VisionObject.video_path = 0
    return redirect(url_for('zones_input'))


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
        converted_filename = new_filename + '.mp4' if new_filename else filename.split('.')[0] + '.mp4'
        converted_file_path = os.path.join(app.config['UPLOAD_FOLDER'], converted_filename)
        file.save(converted_file_path)

        # Delete the original uploaded file
        original_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(original_file_path):
            if original_file_path.split('.')[1] != 'mp4':
                os.remove(original_file_path)
        # Store the temporary frame file path and other data in the session
        VisionObject.video_path = converted_file_path
        return redirect(url_for('zones_input'))
    return 'Invalid file'


@app.route('/zones_input')
def zones_input():
    return render_template('zones_input.html')

@app.route('/image')
def image():
    video_path = VisionObject.video_path
    frame, height, width = get_frame(video_path)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, frame_encoded = cv2.imencode(".jpg", frame, encode_param)
    processed_img_data = base64.b64encode(frame_encoded).decode()
    b64_src = "data:image/jpg;base64,"
    processed_img_data = b64_src + processed_img_data
    return processed_img_data

@app.route('/get_coordinates', methods=['POST'])
def get_coordinates():
    x = int(request.form.get('x'))
    y = int(request.form.get('y'))
    return f'Clicked at x: {x}, y: {y}'


@app.route('/create_zones', methods=['POST'])
def create_zones():
    global polygons
    global polygons_np
    polygons = request.json
    print_polygons(polygons)

    polygons_np = []
    for polygon_data in polygons:
        polygon_coords = [(point['x'], point['y']) for point in polygon_data]
        np_polygon = np.array(polygon_coords, dtype=np.int32)
        polygons_np.append(np_polygon)
    return "Successfully created zones."

@app.route('/generate_zone_frame')
def generate_zone_frame():
    global polygons_np
    if not polygons_np:
        # Handle the case where polygons are not defined
        return "No polygons defined"

    video_path = VisionObject.video_path
    frame, all_zones, zone_annotators, box_annotators = setup_zones(video_path, polygons_np)


    VisionObject.all_zones = all_zones
    VisionObject.box_annotators = box_annotators
    VisionObject.zone_annotators = zone_annotators

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, frame_encoded = cv2.imencode(".jpg", frame, encode_param)
    processed_img_data = base64.b64encode(frame_encoded).decode()
    b64_src = "data:image/jpg;base64,"
    processed_zone_img_data = b64_src + processed_img_data
    return processed_zone_img_data


def print_polygons(polygons):
    for i, zone in enumerate(polygons, start=1):
        print(f'\nZone {i}:')
        for point in zone:
            print(f'\t({point["x"]}, {point["y"]})')

@app.route('/zones')
def zones():
    return render_template('zones_display.html')


def stream_vision(generator, all_zones, zone_annotators, box_annotators, model):
    for frame in generator:
        frame = process_frame(frame, all_zones, zone_annotators, box_annotators, model)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        result, frame_encoded = cv2.imencode(".jpg", frame, encode_param)
        frame = frame_encoded.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cv2.destroyAllWindows()


@app.route('/start_vision')
def start_vision():
    model = VisionObject.model
    video_path = VisionObject.video_path
    all_zones = VisionObject.all_zones
    zone_annotators = VisionObject.zone_annotators
    box_annotators = VisionObject.box_annotators
    generator = get_video_frames_generator(video_path)
    return Response(stream_vision(generator, all_zones, zone_annotators, box_annotators, model), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/crowdvision_zone')
def crowdvision_zone_display():
    return render_template('CrowdVision_Zone_Display.html')


@app.route('/restart', methods=['GET'])
def restart():
    shutil.rmtree(UPLOAD_FOLDER)
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
    return redirect(url_for('main_page'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6942, debug=True)