Instructions for using Facial Recognition


Pipelines ---
    |----->Training Pipeline
            |---->Video Input
            |---->Frames extraction
            |---->Image Augmentation and Preproc
            |---->Model Training
                  |---->Model Registry at Model_Garden
                  |---->Logs in Traininglogs

    |----->Prediction Pipeline
            |---->Select Input Source and Prediction Model
            |---->Frames extraction (source specific)
            |---->Preprocessing
            |---->Prediction using model



-------------------------------------Training and Predictions-----------------------------------------------------------
## Initial Data
1. Record videos of targets' faces from various angles and expressions.
2. Place the video records in 'Data/Facial_Recog/Raw_DataStore/FacialRecog_TargetVideo' folder
3. Rename the video files as '<Target-Name>.mp4'

## Training
1. Update the list of targets in 'Facial_Recognition/TrainingPipeline.py' file with the names of the targets in lowercase.
2. Run the Python 'TrainingPipeline.py'
3. It would generate a trained model in the 'Data/Trained_Model_Garden' folder with the format 'FaceRecogXXXXXXXXXX.hdf5'

## Prediction
1. Run the 'Facial_Recognition/PredictionPipeline.py' file with the following arguments --
        ---For Image
                |----> 1. "image"
                |----> 2. Name of trained model (eg - FaceRecogXXXXX69420.hdf5)
                |----> 3. Path of Image File

        ---For Live
                |----> 1. "camera"
                |----> 2. Name of trained model (eg - FaceRecogXXXXX69420.hdf5)

        ---For Video
                |----> 1. "video"
                |----> 2. Name of trained model (eg - FaceRecogXXXXX69420.hdf5)
                |----> 3. Path of VideoFile

Training Logs saved in ---> 'Facial_Recognition/Training/Traininglogs'
------------------------------------------------------------------------------------------------------------------------

After FTP Integration----
    No data is stored locally at start
    Downloads Target Videos first for training
    Uploads trained Model to server storage
    Removes all training Data from local storage

    Downloads trained model from server and runs prediction

