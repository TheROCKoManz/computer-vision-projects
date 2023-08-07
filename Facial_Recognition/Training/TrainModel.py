import matplotlib.pyplot as plt
from keras.models import Model
import os
from keras.layers import Flatten, Dense
import keras.losses
from keras.applications.inception_resnet_v2 import InceptionResNetV2 as IncRes
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import datetime
import pickle
from Server_Loading.Upload_from_local import upload_files
from Database_Connect.RecordLogs import Register_Model

current_time = datetime.datetime.now()
time_stamp = current_time.strftime("%d%m%y%H%M%S")

def train(Data):

    print("\n\n\n")
    print("Model is being trained on the provided user Images...\n\n\n")

    epochs = 20

    train_data = Data['Train']
    val_data = Data['Val']

    with tf.device('/GPU:0'):
        base_model = IncRes(input_shape=(256,256,3),weights='imagenet',include_top=False)
        for layer in base_model.layers:
            layer.trainable=False

        X=Flatten()(base_model.output)
        X=Dense(units=256, activation='relu')(X)
        X=Dense(units=len(os.listdir("Data/Facial_Recog/ModelData/Train")), activation='softmax')(X)
        model_IncRes = Model(base_model.input, X)
        model_IncRes.compile(optimizer='adam',loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])

        current_time = datetime.datetime.now()
        time_stamp = current_time.strftime("%d%m%y%H%M%S")

        mcIncRes= ModelCheckpoint(filepath="Data/Trained_Model_Garden/FaceRecog"+str(time_stamp)+".hdf5", monitor="val_accuracy", verbose=1, save_best_only= True)
        cbIncRes=[mcIncRes]

        his_IncRes = model_IncRes.fit(train_data, steps_per_epoch=10, epochs=epochs, validation_data=val_data,
                                            validation_steps=8, callbacks=cbIncRes)

    labels = Data['classes']
    pickle_out = open("Data/Trained_Model_Garden/FaceRecog"+str(time_stamp)+".pickle", "wb")
    pickle.dump(labels, pickle_out)
    pickle_out.close()

    training_accuracy = his_IncRes.history['accuracy'][-1]
    training_loss = his_IncRes.history['loss'][-1]
    validation_accuracy = his_IncRes.history['val_accuracy'][-1]
    validation_loss = his_IncRes.history['val_loss'][-1]

    print(f'\n\nTraining Completed...\nEpochs: {epochs}')
    print(f'Training Accuracy: {training_accuracy}\nValidation Accuracy: {validation_accuracy}')
    print(f'Training Loss: {training_loss}\nValidation Loss: {validation_loss}\n')
    print(f' Trained model =====> FaceRecog{str(time_stamp)}.hdf5\n')

    # Close the GPU session
    tf.compat.v1.keras.backend.get_session().close()

    save_training_performance(his_IncRes)

    ModelID = 'FaceRecog'+str(time_stamp)
    Targets = labels
    timestamp = str(current_time)
    accuracy = validation_accuracy
    loss = validation_loss

    Register_Model(modelID=ModelID,
                   targets=Targets,
                   timestamp=timestamp,
                   accuracy=accuracy,
                   loss=loss)

    upload_files('Data/Trained_Model_Garden/')

def save_training_performance(his):
    hIncRes = his.history
    fig = plt.figure(figsize=(17, 5))
    plt.axis("off")
    plt.title(f"ACCURACY vs LOSS -- Model_FaceRecog_{str(time_stamp)}\n\n")
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
    plt.savefig("Facial_Recognition/Training/Traininglogs/training_performace"+str(time_stamp))
