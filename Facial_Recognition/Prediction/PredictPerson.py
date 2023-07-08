from keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input

from ..DataPreproc import Preprocess
import cv2

def predict(frame,modelfile):
    with tf.device('/GPU:0'):
        model = load_model("Data/Trained_Model_Garden/"+modelfile)
    val_path = "Data/ModelData/Val/"
    val_data = Preprocess.preprocessingVal(val_path)
    classes = val_data.class_indices
    img= cv2.resize(frame, (256, 256))
    i = image.img_to_array(img)
    i = preprocess_input(i)
    input_arr = np.array([i])
    pred = np.argmax(model.predict(input_arr))
    pred = list(classes.keys())[pred]
    return pred