import cv2 as cv
import supervision as sv
import os
# USAGE----------------------------------------------------------------
# VideoCapture(): INPUTS Params
#           name ---------> frame-prefix
#           input_path ---> full path of video feed
#           save_path ----> full path to save frames
#           time_limit ---> limit time for clip
#----------------------------------------------------------------------

def face_select(img):
    facedetect = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = facedetect.detectMultiScale(img, 1.1, 5)
    imgs = []

    if len(faces) > 0:
        for x, y, w, h in faces:
            confidence_threshold = 0.7  # Desired minimum face detection confidence in percentage (e.g., 70%)
            # Perform face detection and calculate the confidence
            faces_detected = facedetect.detectMultiScale(img[y:y + h, x:x + w], 1.1, 5, minSize=(30, 30))
            face_confidence = len(faces_detected)
            if len(faces_detected) > confidence_threshold:
                # Calculate the face detection confidence
                confidence = (face_confidence) / len(faces_detected)
                if confidence >= confidence_threshold:
                    imgs.append(img[y:y + h, x:x + w])

    return imgs

def Video2FaceFrames(name, input_path, save_path, time_limit):
    # ---> Frame-name suffix
    counter = len(os.listdir(save_path))+1
    target_name = name
    videoinfo = sv.VideoInfo.from_video_path(input_path)
    vid_resol = (videoinfo.width,videoinfo.height)
    fps = videoinfo.fps
    skip_rate = int(fps)
    video_generator = sv.get_video_frames_generator(input_path)
    for frame in video_generator:
        if time_limit >= 0:
            frame = cv.resize(frame, vid_resol)
            # saving frames
            if counter % skip_rate == 0:
                for img in face_select(frame):
                    print(save_path + "/" + target_name +'_'+ str(counter) + r".jpg")
                    cv.imwrite(save_path + "/" + target_name +'_'+ str(counter) + r".jpg", img)
            counter += 1  # --->increment suffix counter

            # calculate elapsed time of video to limit time
            if counter % int(fps) == 0:
                time_limit -= 1
        else:
            break


