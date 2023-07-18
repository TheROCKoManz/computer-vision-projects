import numpy as np
from tensorflow.keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input
import cv2

def predict(frame,model,classes):
    img= cv2.resize(frame, (256, 256))
    i = image.img_to_array(img)
    i = preprocess_input(i)
    input_arr = np.array([i])
    pred = np.argmax(model.predict(input_arr))
    pred = list(classes.keys())[pred]

    return pred