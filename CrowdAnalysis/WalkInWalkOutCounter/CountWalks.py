import os
HOME = os.getcwd()

import ultralytics
ultralytics.checks()

from utils.install_requirements import pre_setup

pre_setup()

import yolox
print("yolox.__version__:", yolox.__version__)

from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


from supervision.draw.color import ColorPalette
from supervision import Point
from supervision import VideoInfo
from supervision import get_video_frames_generator
from supervision import VideoSink
from supervision.notebook.utils import show_frame_in_notebook
from supervision import Detections, BoxAnnotator
from supervision.detection.line_counter import LineZone, LineZoneAnnotator

from typing import List

import numpy as np


# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks( detections: Detections, tracks: List[STrack]) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids

vid = "/home/manz/Desktop/My Stuffs/Work Stuffs/Python Envs/HyperSpace/ComputerVision/Data/Crowd_Count/Sample_Videos/JapanStreet.mp4"
MODEL = "yolov8x.pt"
from ultralytics import YOLO

model = YOLO(MODEL)
model.fuse()

import supervision as sv
import cv2

# extract video frame
generator = sv.get_video_frames_generator(vid)
iterator = iter(generator)
frame = next(iterator)

# detect
results = model.predict(frame, imgsz=1280)[0]
detections = sv.Detections.from_yolov8(results)
detections = detections[detections.class_id==0]

# annotate
box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1, text_padding = 1)
labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, confidence, class_id, _ in detections]
frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

points = []
cv2.imshow("Image", frame)

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", frame)
        if len(points) == 2:
            cv2.destroyAllWindows()

cv2.setMouseCallback("Image", click_event)

cv2.waitKey(0)

cv2.imwrite(HOME+"/frame.png",frame)

# settings
LINE_START = Point(1346,1078)
LINE_END = Point(630,572)

VideoInfo.from_video_path(vid)


# create BYTETracker instance
byte_tracker = BYTETracker(BYTETrackerArgs())
# create VideoInfo instance
video_info = VideoInfo.from_video_path(vid)
# create frame generator
generator = get_video_frames_generator(vid)
# create LineCounter instance
line_counter = LineZone(start=LINE_START, end=LINE_END)
# create instance of BoxAnnotator and LineCounterAnnotator
box_annotator = BoxAnnotator(thickness=2, text_thickness=2, text_scale=1, text_padding = 1)
line_annotator = LineZoneAnnotator(thickness=3, text_thickness=2, text_scale=3, text_padding=1)

# open target video file
for frame in generator:
    # model prediction on single frame and conversion to supervision Detections
    results = model.predict(frame, imgsz=1280)[0]
    detections = sv.Detections.from_yolov8(results)

    # filtering out detections with unwanted classes
    detections = detections[detections.class_id == 0]

    # tracking detections
    tracks = byte_tracker.update(
        output_results=detections2boxes(detections=detections),
        img_info=frame.shape,
        img_size=frame.shape
    )
    tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)

    detections.tracker_id = np.array(tracker_id)

    # filtering out detections without trackers
    mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
    detections.filter(mask=mask, inplace=True)

    # format custom labels
    labels = [
        f"#{tracker_id} -- {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections
    ]

    # updating line counter
    line_counter.trigger(detections=detections)

    # annotate and display frame
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    line_annotator.annotate(frame=frame, line_counter=line_counter)

    cv2.imshow('Sample_Output', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
