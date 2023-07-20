from ftplib import FTP
from io import BytesIO
import cv2
from Prediction.PredictPerson import predict
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf
import sys
import cv2
import pickle
import collections
import os
import time
import calendar

current_GMT = time.gmtime()
time_stamp = calendar.timegm(current_GMT)

server_ip = '144.76.182.206'
username = 'computervision'
password = 'Computervision@253#87'
data_dir = '/ComputerVision/'

ftp = FTP(host=server_ip, user=username, passwd=password)
ftp.cwd(data_dir)


def predictImagefile(imgpath, modelfile, labels):
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    font = cv2.FONT_HERSHEY_COMPLEX

    core_config = tf.compat.v1.ConfigProto()
    core_config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=core_config)
    tf.compat.v1.keras.backend.set_session(session)

    model_stream = BytesIO()
    labels_stream = BytesIO()
    ftp.retrbinary('RETR ' + modelfile, model_stream.write)
    ftp.retrbinary('RETR ' + labels, labels_stream.write)
    model_stream.seek(0)
    labels_stream.seek(0)

    with open('imgface_model.h5', 'wb') as local_model:
        local_model.write(model_stream.read())
    with open('imglabels.pickle', 'wb') as local_label:
        local_label.write(labels_stream.read())
    local_model.close()
    local_label.close()

    with tf.device('/GPU:0'):
        model = load_model('imgface_model.h5')
    pickle_in = open('imglabels.pickle', "rb")
    classes = pickle.load(pickle_in)
    pickle_in.close()

    os.remove('imgface_model.h5')
    os.remove('imglabels.pickle')

    img = cv2.imread(imgpath)

    faces = facedetect.detectMultiScale(img, 1.1, 5)
    for x, y, w, h in faces: cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    pred = predict(frame=img, model=model, classes=classes)

    cv2.putText(img, pred, (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow('Image', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    ftp.close()

    return f"\n\nFinal Recognized: {pred}"


def predictVideofile(vidpath, modelfile, labels):
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(vidpath)
    font = cv2.FONT_HERSHEY_COMPLEX

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)
    detection = []
    # result = cv2.VideoWriter('Facial_Recognition/Prediction/Predictionlogs/prediction_'+str(time_stamp)+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, size)

    core_config = tf.compat.v1.ConfigProto()
    core_config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=core_config)
    tf.compat.v1.keras.backend.set_session(session)

    model_stream = BytesIO()
    labels_stream = BytesIO()
    ftp.retrbinary('RETR ' + modelfile, model_stream.write)
    ftp.retrbinary('RETR ' + labels, labels_stream.write)
    model_stream.seek(0)
    labels_stream.seek(0)

    with open('vidface_model.h5', 'wb') as local_model:
        local_model.write(model_stream.read())
    with open('vidlabels.pickle', 'wb') as local_label:
        local_label.write(labels_stream.read())
    local_model.close()
    local_label.close()

    with tf.device('/GPU:0'):
        model = load_model('vidface_model.h5')
    pickle_in = open('vidlabels.pickle', "rb")
    classes = pickle.load(pickle_in)
    pickle_in.close()

    os.remove('vidface_model.h5')
    os.remove('vidlabels.pickle')

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        faces = facedetect.detectMultiScale(frame, 1.1, 5)
        for x, y, w, h in faces: cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        pred = predict(frame=frame, model=model, classes=classes)
        print(pred)
        detection.append(pred)
        cv2.putText(frame, pred, (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Webcam', frame)
        # result.write(frame)
        # if detection.__len__() == 15:
        #     break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    ftp.close()
    Recognized_Person = collections.Counter(detection).most_common(1)[0][0]
    return f"\n\nFinal Recognized: {Recognized_Person}"


def predictLive(modelfile, labels):
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # cap = cv2.VideoCapture(1)  #Priority on external Cam
    cap = cv2.VideoCapture(0)
    # Check if the camera is opened
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_COMPLEX

    core_config = tf.compat.v1.ConfigProto()
    core_config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=core_config)
    tf.compat.v1.keras.backend.set_session(session)

    model_stream = BytesIO()
    labels_stream = BytesIO()
    ftp.retrbinary('RETR ' + modelfile, model_stream.write)
    ftp.retrbinary('RETR ' + labels, labels_stream.write)
    model_stream.seek(0)
    labels_stream.seek(0)

    with open('liveface_model.h5', 'wb') as local_model:
        local_model.write(model_stream.read())
    with open('livelabels.pickle', 'wb') as local_label:
        local_label.write(labels_stream.read())
    local_model.close()
    local_label.close()

    with tf.device('/GPU:0'):
        model = load_model('liveface_model.h5')
    pickle_in = open('livelabels.pickle', "rb")
    classes = pickle.load(pickle_in)
    pickle_in.close()

    os.remove('liveface_model.h5')
    os.remove('livelabels.pickle')

    detection = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        faces = facedetect.detectMultiScale(frame, 1.1, 5)
        if len(faces) > 0:
            for x, y, w, h in faces: cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            pred = predict(frame=frame, model=model, classes=classes)
            print(pred)
            detection.append(pred)
            cv2.putText(frame, pred, (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Webcam', frame)

        if detection.__len__() == 15:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    ftp.close()

    Recognized_Person = collections.Counter(detection).most_common(1)[0][0]
    return f"\n\nFinal Recognized: {Recognized_Person}"


if __name__ == '__main__':
    source = sys.argv[1]
    modelfile = 'Data/Trained_Model_Garden/' + sys.argv[2]
    if modelfile[-5:] != '.hdf5':
        modelfile = modelfile + '.hdf5'

    labels = modelfile[:-5] + '.pickle'

    if source == 'image':
        img_path = sys.argv[3]
        print(predictImagefile(img_path, modelfile, labels))

    elif source == 'video':
        vid_path = sys.argv[3]
        print(predictVideofile(vid_path, modelfile, labels))

    elif source == 'camera':
        print(predictLive(modelfile, labels))
