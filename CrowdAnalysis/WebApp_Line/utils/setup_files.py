import os
import sys
import subprocess
from ultralytics import YOLO

HOME=os.getcwd()

def run_setup_develop():
    try:
        print(os.getcwd())
        subprocess.run(['pip', 'install', '-e', '.'], check=True)
        print("Setup complete.")
    except subprocess.CalledProcessError as e:
        print("Error during setup:", e)

def Model():
    MODEL = "yolov8x.pt"
    model = YOLO(MODEL)
    model.fuse()
    return model

# Usage
def pre_setup():
    project_home = HOME + '/CrowdAnalysis/WebApp_Line'
    os.chdir(project_home)
    if not os.path.exists(project_home+'/ByteTrack'):
        repo_url = "https://github.com/ifzhang/ByteTrack.git"
        command = ["git", "clone", repo_url]
        try:
            subprocess.run(command, check=True)
            print("ByteTrack cloned successful")
        except subprocess.CalledProcessError:
            print("ByteTrack cloned failed")
    else:
        print("ByteTrack already present")

    os.chdir(HOME+'/CrowdAnalysis/WalkInWalkOutCounter/ByteTrack/')
    sys.path.append(f"{HOME}/CrowdAnalysis/WalkInWalkOutCounter/ByteTrack")
    run_setup_develop()
    os.chdir(project_home)