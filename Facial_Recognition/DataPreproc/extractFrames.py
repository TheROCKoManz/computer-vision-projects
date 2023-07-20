import os

import DataPreproc.Video2Frames as V2F


def extractFrames():
    print("\nExtracting Frames from Videos...")

    raw_origin_path = 'Data/Facial_Recog/Raw_DataStore/FacialRecog_TargetVideo/'
    frames_destination_path = 'Data/Facial_Recog/Raw_DataStore/FacialRecog_TargetFrames/'

    for target in [file[:-4] for file in os.listdir(raw_origin_path) if file not in ['.gitkeep', '.gitignore']]:
        if not os.path.exists(frames_destination_path):
            os.mkdir(frames_destination_path)
        if not os.path.exists(frames_destination_path+target.lower()):
            os.mkdir(frames_destination_path+target.lower())

            V2F.Video2Frames(name=target.lower(),
                             input_path=raw_origin_path+target+'.mp4',
                             save_path=frames_destination_path+target.lower(),
                             time_limit=60)

    print("Frames Extracted\n")
