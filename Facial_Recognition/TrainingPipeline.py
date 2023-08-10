import argparse
import os
import shutil

from DataPreproc import Preprocess, extractFrames
from Server_Loading import Download_from_Server as DFS
from Training import TrainModel


def delete_localfiles(delt = 'all'):
    image_base_dir = '_'
    work_dir = '_'
    ModelData_dir = '_'
    if delt == 'all':
        image_base_dir = 'Data/Facial_Recog/Raw_DataStore/FacialRecog_TargetFrames/'
        work_dir = 'Data/Facial_Recog/Preproc_Data/'
        ModelData_dir = 'Data/Facial_Recog/ModelData/'
    else:
        if delt == 'image_base_dir':
            image_base_dir = 'Data/Facial_Recog/Raw_DataStore/FacialRecog_TargetFrames/'
        if delt == 'work_dir':
            work_dir = 'Data/Facial_Recog/Preproc_Data/'
        if delt == 'ModelData_dir':
            ModelData_dir = 'Data/Facial_Recog/ModelData/'

    for dir in os.listdir(work_dir):
        if dir not in ['.gitkeep', '.gitignore'] and os.path.exists(work_dir+dir):
            shutil.rmtree(work_dir+dir)
    for dir in os.listdir(ModelData_dir):
        if dir not in ['.gitkeep', '.gitignore'] and os.path.exists(ModelData_dir+dir):
            shutil.rmtree(ModelData_dir+dir)
    if os.path.exists(image_base_dir):
        shutil.rmtree(image_base_dir)

def train(Targets):
    print('\n\nTraining_Mode_ON\n')
    delete_localfiles()
    DFS.download_videos(Targets)
    extractFrames.extractFrames(Targets)
    delete_localfiles('image_base_dir')
    Data = Preprocess.preprocess(Targets)
    delete_localfiles('work_dir')
    Model_ID = TrainModel.train(Data)
    delete_localfiles()
    return Model_ID


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a list of target users.")
    parser.add_argument("--targets", type=str, help='List of target users, e.g., "[user_001, user_002, user4, user_006, ...]"')
    args = parser.parse_args()

    targets = args.targets.strip('[]').lower().split(', ')

    train(targets)
