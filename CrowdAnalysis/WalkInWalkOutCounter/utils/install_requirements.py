import os
import sys
import subprocess
HOME=os.getcwd()


def run_setup_develop():
    try:
        print(os.getcwd())
        subprocess.run(['pip', 'install', '-e', '.'], check=True)
        print("Setup complete.")
    except subprocess.CalledProcessError as e:
        print("Error during setup:", e)


# Usage
def pre_setup():
    os.chdir(HOME+'/CrowdAnalysis//WalkInWalkOutCounter/ByteTrack/')
    sys.path.append(f"{HOME}/CrowdAnalysis/WalkInWalkOutCounter/ByteTrack")
    run_setup_develop()
    os.chdir(HOME + '/CrowdAnalysis//WalkInWalkOutCounter/')