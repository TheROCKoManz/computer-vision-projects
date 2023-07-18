from Prediction.PredictPerson import predict
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf
import sys
import cv2
import pickle
import collections

import time
import calendar
current_GMT = time.gmtime()
time_stamp = calendar.timegm(current_GMT)


def predictImagefile(imgpath, modelfile, labels):
    img = image.load_img(imgpath, target_size=(256, 256, 3))
    prediction = predict(frame=img, modelfile=modelfile, labels = labels)
    return prediction


def predictVideofile(vidpath, modelfile, labels):
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(vidpath)
    font = cv2.FONT_HERSHEY_COMPLEX

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)
    detection = []
    # result = cv2.VideoWriter('Facial_Recognition/Prediction/Predictionlogs/prediction_'+str(time_stamp)+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, size)

    while True:
        ret, frame = cap.read()
        faces = facedetect.detectMultiScale(frame, 1.1, 5)
        for x, y, w, h in faces: cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        core_config = tf.compat.v1.ConfigProto()
        core_config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=core_config)
        tf.compat.v1.keras.backend.set_session(session)

        with tf.device('/GPU:0'):
            model = load_model(modelfile)

        pickle_in = open(labels, "rb")
        classes = pickle.load(pickle_in)

        pred = predict(frame=frame,model=model, classes= classes)
        print(pred)
        detection.append(pred)
        cv2.putText(frame, pred, (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Webcam', frame)
        # result.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
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
    with tf.device('/GPU:0'):
        model = load_model(modelfile)

    pickle_in = open(labels, "rb")
    classes = pickle.load(pickle_in)
    detection = []
    while True:
        ret, frame = cap.read()
        faces = facedetect.detectMultiScale(frame, 1.1, 5)
        if len(faces)>0:
            for x, y, w, h in faces: cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            pred = predict(frame=frame,model=model, classes= classes)
            print(pred)
            detection.append(pred)
            cv2.putText(frame, pred, (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Webcam', frame)

        if detection.__len__()==15:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    Recognized_Person = collections.Counter(detection).most_common(1)[0][0]
    return f"\n\nFinal Recognized: {Recognized_Person}"


if __name__ == '__main__':
    source = sys.argv[1]
    modelfile = 'Data/Trained_Model_Garden/'+sys.argv[2]
    if modelfile[-5:]!='.hdf5':
        modelfile = modelfile+'.hdf5'

    labels = modelfile[:-5]+'.pickle'

    if source == 'image':
        img_path = sys.argv[3]
        predictImagefile(img_path, modelfile, labels)

    elif source == 'video':
        vid_path = sys.argv[3]
        print(predictVideofile(vid_path, modelfile, labels))

    elif source == 'camera':
        print(predictLive(modelfile, labels))

