This system counts the number of people moving from one zone to another.
Object Detection on Person class + Line trigger to count in and outbounds.

Usage----------------------------------------------------------------------
Open Terminal-----------------
1. Navigate cwd to WalkInWalkOutCounter directory.
2. git clone https://github.com/ifzhang/ByteTrack.git
3. cd ByteTrack
4. pip3 install -q -r requirements.txt
5. pip install -e .
6. pip install -q cython_bbox onemetric loguru lap thop
7. pip install numpy==1.22.4
----------------------------------------------------------------------------

Run CrowdWalkRun.py------------
Syntax: python CrowdWalkRun.py <source> <filepath:optional>

Arguments:
<source> : camera (for live feed), file (for loading recorded video)
<filepath:optional> : path of video file if source is 'file'
----------------------------------------------------------------------------

Object Detection model used ----> Yolov8x
Object Tracking ----------------> ByteTrack
-----------------------------------------------------------------------------