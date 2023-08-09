import os
import supervision as sv
import sys
sys.path.append('Facial_Recognition/')
sys.path.append('ComputerVision/')
import DataPreproc.Video2Frames as V2F


def extractFrames(Targets):
    print("\nExtracting Frames from Videos...")

    raw_origin_path = 'Data/Facial_Recog/Raw_DataStore/FacialRecog_TargetVideo/'
    frames_destination_path = 'Data/Facial_Recog/Raw_DataStore/FacialRecog_TargetFrames/'
    if not os.path.exists(frames_destination_path):
        os.mkdir(frames_destination_path)
    for file in os.listdir(raw_origin_path):
        if file not in ['.gitkeep', '.gitignore']:
            target, _ = os.path.splitext(file)  # Split filename and extension
            user_id = target.lower().split('_')[0] # Extract user ID from filename
            if user_id in Targets:
                target_folder = os.path.join(frames_destination_path, user_id)
                if not os.path.exists(target_folder):
                    os.mkdir(target_folder)
                    V2F.Video2Frames(name=user_id,
                                     input_path=raw_origin_path + target + '.mp4',
                                     save_path=frames_destination_path + user_id,
                                     time_limit=60)

    print("Frames Extracted\n")
