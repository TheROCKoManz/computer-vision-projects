from Prediction.PredictPerson import predict
from tensorflow.keras.preprocessing import image
import sys

if __name__ == '__main__':
    img = sys.argv[1]
    modelfile = sys.argv[2]
    img = image.load_img(img, target_size=(256, 256, 3))
    predict(image,modelfile)