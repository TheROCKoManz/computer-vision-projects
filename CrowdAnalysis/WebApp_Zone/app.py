import base64
import shutil
import sys
import os
sys.path.append('CrowdAnalysis/WebApp_Zone/')
import numpy as np
import cv2
from flask import Blueprint, Flask, render_template, request, redirect, url_for, Response, current_app, jsonify
from werkzeug.utils import secure_filename
from CrowdZonesCount import get_frame, setup_zones, process_frame, Model
# app = Flask(__name__)
from supervision import get_video_frames_generator

# Define the blueprint
zone_bp = Blueprint('zone_bp', __name__, template_folder='templates')

UPLOAD_FOLDER = 'Data/Crowd_Count/ZoneCounter_Dynamic/uploads'
ALLOWED_EXTENSIONS = {'avi', 'mp4', 'mov', 'wmv', 'flv', 'mkv', 'webm', 'mpeg', '3gp', 'ts', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

zone_bp.secret_key = generate_secret_key()
# zone_bp.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def create_upload_folder():
    upload_folder = current_app.config.get('UPLOAD_FOLDER')
    if upload_folder and not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@zone_bp.route('/')
def main_page():
    return render_template('zonalcrowdcount.html')

@zone_bp.route('/get_url/<endpoint>')
def get_url(endpoint):
    return jsonify(url=url_for(f'zone_bp.{endpoint}'))

@zone_bp.route('/camera')
def camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Scope for Camera Access"
    cap.release()
    VisionObject.video_path = 0
    print("DEBUG: ", url_for('zone_bp.zones_input'))
    return redirect(url_for('zone_bp.zones_input'))

@zone_bp.route('/upload', methods=['POST'])
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
        converted_file_path = os.path.join(UPLOAD_FOLDER, converted_filename)
        file.save(converted_file_path)

        # Delete the original uploaded file
        original_file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(original_file_path):
            if original_file_path.split('.')[1] != 'mp4':
                os.remove(original_file_path)
        # Store the temporary frame file path and other data in the session
        VisionObject.video_path = converted_file_path
        print("DEBUG: ",url_for('zone_bp.zones_input'))
        return redirect(url_for('zone_bp.zones_input'))
    return 'Invalid file'

@zone_bp.route('/zones_input')
def zones_input():
    return render_template('zones_input.html')

@zone_bp.route('/image')
def image():
    video_path = VisionObject.video_path
    frame, height, width = get_frame(video_path)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, frame_encoded = cv2.imencode(".jpg", frame, encode_param)
    processed_img_data = base64.b64encode(frame_encoded).decode()
    b64_src = "data:image/jpg;base64,"
    processed_img_data = b64_src + processed_img_data
    return processed_img_data

@zone_bp.route('/get_coordinates', methods=['POST'])
def get_coordinates():
    x = int(request.form.get('x'))
    y = int(request.form.get('y'))
    return f'Clicked at x: {x}, y: {y}'

@zone_bp.route('/create_zones', methods=['POST'])
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
    return {"msg":"Successfully created zones."}

@zone_bp.route('/generate_zone_frame')
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

@zone_bp.route('/zones')
def zones():
    print("Here")
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

@zone_bp.route('/start_vision')
def start_vision():
    model = VisionObject.model
    video_path = VisionObject.video_path
    all_zones = VisionObject.all_zones
    zone_annotators = VisionObject.zone_annotators
    box_annotators = VisionObject.box_annotators
    generator = get_video_frames_generator(video_path)
    return Response(stream_vision(generator, all_zones, zone_annotators, box_annotators, model), mimetype='multipart/x-mixed-replace; boundary=frame')

@zone_bp.route('/crowdvision_zone')
def crowdvision_zone_display():
    return render_template('CrowdVision_Zone_Display.html')

@zone_bp.route('/restart', methods=['GET'])
def restart():
    shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    print("DEBUG: ", url_for('zone_bp.main_page'))
    return redirect(url_for('zone_bp.main_page'))

# Function to create the app
def create_zone_blueprint():
    return zone_bp

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=6942, debug=True)