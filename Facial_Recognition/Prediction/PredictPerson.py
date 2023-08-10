import numpy as np
from tensorflow.keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input
import cv2

def predict(frame,model,classes):
    img= cv2.resize(frame, (256, 256))
    i = image.img_to_array(img)
    i = preprocess_input(i)
    input_arr = np.array([i])
    pred = model.predict(input_arr)
    predicted_class = np.argmax(pred, axis=1)[0]
    prediction_accuracy = pred[0][predicted_class]

    pred = np.argmax(pred)
    pred_result = list(classes.keys())[pred]

    return pred_result, prediction_accuracy