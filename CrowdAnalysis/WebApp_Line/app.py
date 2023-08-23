#### TEST VISION........###
import os
HOME=os.getcwd()
project_home = HOME + '/CrowdAnalysis/WebApp_Line'
# from utils import setup_files
# setup_files.pre_setup()
os.chdir(HOME)
import base64
import shutil
import cv2
from flask import Flask, render_template, request, redirect, url_for, Response
from werkzeug.utils import secure_filename
from CountWalks import Model, get_frame, process_frame, setup_lines
app = Flask(__name__)
from supervision import get_video_frames_generator
UPLOAD_FOLDER = 'Data/Crowd_Count/LineCounter_Dynamic/uploads'
ALLOWED_EXTENSIONS = {'avi', 'mp4', 'mov', 'wmv', 'flv', 'mkv', 'webm', 'mpeg', '3gp', 'ts', 'gif'}

class LineVision:
    def __init__(self):
        self.video_path = ''
        self.byte_tracker = []
        self.box_annotator = []
        self.line_annotators = []
        self.line_counters = []
        self.model = Model()


VisionObject = LineVision()


def generate_secret_key(length=32):
    return os.urandom(length).hex()


app.secret_key = generate_secret_key()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def main_page():
    os.chdir(HOME)
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)
    return render_template('linecrowdcount.html')

@app.route('/camera')
def camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Scope for Camera Access"
    cap.release()
    VisionObject.video_path = 0
    return redirect(url_for('lines_input'))




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
        return redirect(url_for('lines_input'))
    return 'Invalid file'


@app.route('/lines_input')
def lines_input():
    return render_template('lines_input.html')

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


@app.route('/create_lines', methods=['POST'])
def create_lines():
    global lines, lines_json
    lines_json = request.json
    print_lines(lines_json)

    lines = []
    for line in lines_json:
        line_ends = [(point['x'], point['y']) for point in line]
        lines.append(line_ends)
    return "Successfully created lines."

@app.route('/generate_line_frame')
def generate_line_frame():
    global lines
    if not lines:
        # Handle the case where polygons are not defined
        return "No polygons defined"

    video_path = VisionObject.video_path
    print('\n\nLines')
    print(lines)

    frame, byte_tracker, box_annotator, line_annotators, line_counters = setup_lines(video_path, lines)


    VisionObject.byte_tracker = byte_tracker
    VisionObject.box_annotator = box_annotator
    VisionObject.line_annotators = line_annotators
    VisionObject.line_counters = line_counters

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, frame_encoded = cv2.imencode(".jpg", frame, encode_param)
    processed_img_data = base64.b64encode(frame_encoded).decode()
    b64_src = "data:image/jpg;base64,"
    processed_line_img_data = b64_src + processed_img_data
    return processed_line_img_data


def print_lines(lines):
    for i, line in enumerate(lines, start=1):
        print(f'\nLine {i}:')
        for point in line:
            print(f'\t({point["x"]}, {point["y"]})')

@app.route('/lines')
def lines():
    return render_template('lines_display.html')


def stream_vision(generator, byte_tracker, line_counters, box_annotator, line_annotators, model):
    for frame in generator:
        frame = process_frame(frame, byte_tracker, line_counters, box_annotator, line_annotators, model)

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
    byte_tracker = VisionObject.byte_tracker
    line_counters = VisionObject.line_counters
    box_annotator = VisionObject.box_annotator
    line_annotators = VisionObject.line_annotators
    generator = get_video_frames_generator(video_path)
    return Response(stream_vision(generator, byte_tracker, line_counters, box_annotator, line_annotators, model), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/crowdvision_line')
def crowdvision_line():
    return render_template('CrowdVision_Line_Display.html')


@app.route('/restart', methods=['GET'])
def restart():
    shutil.rmtree(UPLOAD_FOLDER)
    return redirect(url_for('main_page'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6943, debug=True)