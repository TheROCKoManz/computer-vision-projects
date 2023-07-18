import numpy as np
import matplotlib.pyplot as plt
import os
import math
import shutil
from keras.preprocessing.image import ImageDataGenerator

image_base_dir = 'Data/Facial_Recog/Raw_DataStore/FacialRecog_TargetFrames/'
work_dir = 'Data/Facial_Recog/Preproc_Data/'
ModelData_dir = 'Data/Facial_Recog/ModelData/'

no_of_images = {}

def data_modelling(Targets):
    Non_Targets = []
    for person in os.listdir(image_base_dir):
        if person not in Targets and person not in ['.gitkeep', '.gitignore']:
            Non_Targets.append(person)
    personfolder = os.listdir(image_base_dir)

    if not os.path.exists(work_dir):
        os.mkdir(work_dir)
    if not os.path.exists(work_dir + 'Targets'):
        os.mkdir(work_dir + 'Targets')
    # if not os.path.exists(work_dir + 'Non_Targets'):
    #     os.mkdir(work_dir + 'Non_Targets')

    for target in Targets:
        target_dir = work_dir + 'Targets/' + target
        if target not in os.listdir(work_dir + 'Targets/') and target in personfolder and target not in ['.gitkeep', '.gitignore']:
            os.mkdir(target_dir)

        for file in os.listdir(image_base_dir + target):
            if file not in os.listdir(target_dir) and file not in ['.gitkeep', '.gitignore']:
                shutil.copy(image_base_dir + target + '/' + file, target_dir)

    # for person in Non_Targets:
    #     non_target_dir = work_dir + 'Non_Targets/'
    #     counter = 0
    #     for file in os.listdir(image_base_dir + person):
    #         if file not in os.listdir():
    #             shutil.copy(image_base_dir + person + '/' + file, non_target_dir)
    #         counter += 1
    #         if counter == 5:
    #             break

    for target in os.listdir(work_dir + 'Targets'):
        if target not in ['.gitkeep', '.gitignore']:
            no_of_images[target] = len(os.listdir(work_dir + 'Targets/' + target))
    # no_of_images['Non_Targets'] = len(os.listdir(work_dir + 'Non_Targets/'))

    print('\n\n\nTarget_Name\t\tNo_of_Images')
    for item in Targets:
        print(item, '\t\t', no_of_images[item])
    # print('Non_Targets', '\t\t', no_of_images['Non_Targets'],'\n\n\n')

    return no_of_images

def dataFolder(p, split):
    if not os.path.exists(ModelData_dir):
        os.mkdir(ModelData_dir)
    if not os.path.exists(ModelData_dir + p):
        os.mkdir(ModelData_dir + p)
        for dir in os.listdir(work_dir + 'Targets/'):
            if dir not in ['.gitkeep', '.gitignore']:
                os.makedirs(ModelData_dir + p + "/" + dir)
                for img in np.random.choice(a=os.listdir(os.path.join(work_dir + 'Targets', dir)),
                                            size=(math.floor(split * no_of_images[dir]) - 5),
                                            replace=False):
                    O = os.path.join(work_dir + 'Targets', dir, img)
                    D = os.path.join(ModelData_dir + p, dir)
                    shutil.copy(O, D)
            # os.mkdir(ModelData_dir + p + "/Non_Targets")
            # for img in np.random.choice(a=os.listdir(work_dir + 'Non_Targets/'),
            #                             size=(math.floor(split * no_of_images['Non_Targets']) - 5),
            #                             replace=False):
            #     O = os.path.join(work_dir + 'Non_Targets', img)
            #     D = os.path.join(ModelData_dir + p + '/Non_Targets')
        #     shutil.copy(O, D)

    else:
        print(f"{p} Exists")

def preprocessingTrain(path):
    image_data = ImageDataGenerator(featurewise_center=True,
                                    rotation_range=0.4,
                                    width_shift_range=0.3,
                                    zoom_range=0.2,
                                    shear_range=0.2,
                                    rescale=1./255,
                                    horizontal_flip= True)

    image = image_data.flow_from_directory(directory= path,
                                           target_size=(256,256),
                                           batch_size=12,
                                           class_mode='categorical')
    return image

def preprocessingVal(path):
    image_data = ImageDataGenerator(rescale=1./255)
    image = image_data.flow_from_directory(directory= path,
                                           target_size=(256,256),
                                           batch_size=12,
                                           class_mode='categorical')
    return image


def preprocess(Targets):

    data_modelling(Targets)

    dataFolder("Train", 0.7)
    dataFolder("Val", 0.3)
    dataFolder("Test", 0.5)

    train_path = ModelData_dir+"Train"
    train_data = preprocessingTrain(train_path)

    val_path = ModelData_dir+"Val"
    val_data = preprocessingVal(val_path)

    test_path = ModelData_dir+"Test"
    test_data = preprocessingVal(test_path)

    classes = val_data.class_indices

    Data = {'Train': train_data, 'Test': test_data, 'Val': val_data, 'classes': classes}

    return Data