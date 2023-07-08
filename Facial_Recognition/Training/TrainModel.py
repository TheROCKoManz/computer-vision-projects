import matplotlib.pyplot as plt
from keras.models import Model
import os
from keras.layers import Flatten, Dense
import keras.losses
from keras.applications.inception_resnet_v2 import InceptionResNetV2 as IncRes
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import calendar
import time

current_GMT = time.gmtime()
time_stamp = calendar.timegm(current_GMT)

def train(Data):

    print("\n\n\n")
    print("Model is being trained on the provided user Images...\n\n\n")

    train_data = Data['Train']
    val_data = Data['Val']

    with tf.device('/GPU:0'):
        base_model = IncRes(input_shape=(256,256,3),weights='imagenet',include_top=False)
        for layer in base_model.layers:
            layer.trainable=False

        X=Flatten()(base_model.output)
        X=Dense(units=256, activation='relu')(X)
        X=Dense(units=len(os.listdir("Data/ModelData/Train")), activation='softmax')(X)
        model_IncRes = Model(base_model.input, X)
        model_IncRes.compile(optimizer='adam',loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])

        current_GMT = time.gmtime()
        time_stamp = calendar.timegm(current_GMT)

        mcIncRes= ModelCheckpoint(filepath="FaceRecog"+str(time_stamp)+".hdf5", monitor="val_accuracy", verbose=1, save_best_only= True)
        cbIncRes=[mcIncRes]

        his_IncRes = model_IncRes.fit_generator(train_data, steps_per_epoch=12, epochs=10, validation_data=val_data,
                                            validation_steps=10, callbacks=cbIncRes)
    save_training_performance(his_IncRes)

def save_training_performance(his):
    hIncRes = his.history
    plt.savefig('')
    fig = plt.figure(figsize=(17, 5))
    plt.axis("off")
    plt.title("ACCURACY vs LOSS\n\n")
    rows = 1
    columns = 2
    fig.add_subplot(rows, columns, 1)
    plt.title("ACCURACY")
    plt.plot(hIncRes['accuracy'], c='blue')
    plt.plot(hIncRes['val_accuracy'], c='red')
    fig.add_subplot(rows, columns, 2)
    plt.title("LOSS")
    plt.plot(hIncRes['loss'], c='blue')
    plt.plot(hIncRes['val_loss'], c='red')
    plt.savefig("training_performace")