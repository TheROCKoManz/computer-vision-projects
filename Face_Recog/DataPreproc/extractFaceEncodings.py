import face_recognition
from PIL import Image
import numpy as np
import cv2
import sys
sys.path.append('../')
sys.path.append('Face_Recog')

from DataPreproc import extractFrames
from Server_Loading import Download_from_Server as DFS
import warnings
import json
import os
warnings.filterwarnings("ignore")

def get_average_face_embedding_from_folder(folder_path):
    all_face_embeddings = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Detect facial landmarks
            face_landmarks_list = face_recognition.face_landmarks(image)

            for landmarks in face_landmarks_list:
                # Get facial landmarks
                left_eye = landmarks['left_eye']
                right_eye = landmarks['right_eye']

                # Calculate the center between the eyes for face alignment
                eye_center = np.mean([left_eye[3], right_eye[0]], axis=0)

                # Calculate the rotation angle based on the eyes
                dY = right_eye[0][1] - left_eye[3][1]
                dX = right_eye[0][0] - left_eye[3][0]
                angle = np.degrees(np.arctan2(dY, dX))

                # Rotate and align the face using PIL
                pil_image = Image.fromarray(image)
                aligned_face = pil_image.rotate(angle, center=tuple(eye_center))
                aligned_face = np.array(aligned_face)

                # Compute face encoding for the aligned face
                embedding = face_recognition.face_encodings(aligned_face)
                if len(embedding)>0:
                    all_face_embeddings.append(embedding[0])

    if all_face_embeddings:
        average_embedding = np.mean(all_face_embeddings, axis=0)
        return average_embedding
    else:
        return None


def encode_embedding_to_json(embedding):
    serializable_embedding = embedding.tolist()
    encoded_json = json.dumps(serializable_embedding)
    return encoded_json


def get_embedding_train(Targets):
    DFS.download_videos(Targets)
    extractFrames.extractFrames(Targets)

    target_dir = 'Data/Facial_Recog/Raw_DataStore/FacialRecog_TargetFrames/'+ Targets[0]
    average_embedding = get_average_face_embedding_from_folder(target_dir)
    encoded_json = encode_embedding_to_json(average_embedding)
    return encoded_json


