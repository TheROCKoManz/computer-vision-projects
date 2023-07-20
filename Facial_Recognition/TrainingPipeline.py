from DataPreproc import Preprocess, extractFrames
from Training import TrainModel
from Server_Loading import Download_from_Server as DFS
import os, shutil

def delete_localfiles():
    image_base_dir = 'Data/Facial_Recog/Raw_DataStore/FacialRecog_TargetFrames/'
    work_dir = 'Data/Facial_Recog/Preproc_Data/'
    ModelData_dir = 'Data/Facial_Recog/ModelData/'

    for dir in os.listdir(work_dir):
        if dir not in ['.gitkeep', '.gitignore']:
            shutil.rmtree(work_dir+dir)
    for dir in os.listdir(ModelData_dir):
        if dir not in ['.gitkeep', '.gitignore']:
            shutil.rmtree(ModelData_dir+dir)
    shutil.rmtree(image_base_dir)

def main():
    Targets = ['manasij', 'abhishek'] # list of training targets

    DFS.download_videos(Targets)

    extractFrames.extractFrames()

    Data = Preprocess.preprocess(Targets)
    TrainModel.train(Data)
    delete_localfiles()

if __name__ == '__main__':
    main()
