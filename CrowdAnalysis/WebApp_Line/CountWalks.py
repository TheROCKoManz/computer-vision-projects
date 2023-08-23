import sys
import ultralytics
import yolox
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision import Point
from supervision import get_video_frames_generator
from supervision import Detections, BoxAnnotator
from supervision.detection.line_counter import LineZone, LineZoneAnnotator
from typing import List
import numpy as np
import supervision as sv
import cv2
from screeninfo import get_monitors
import os
import subprocess
from ultralytics import YOLO
#---------------------------------------------------------------------------------------

HOME=os.getcwd()

screen_width = get_monitors()[0].width
screen_height = get_monitors()[0].height

def run_setup_develop():
    try:
        print(os.getcwd())
        subprocess.run(['pip', 'install', '-e', '.', '-q'], check=True)
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


class LINE:
    def __init__(self, frame):
        self.image = frame
        self.Point1 = None
        self.Point2 = None
        self.click_count = 0
        self.click_points = []

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.click_count < 2:
            self.click_points.append((x, y))
            self.click_count += 1
            cv2.circle(self.image, (x, y), 5, (0, 255, 255), -1)  # Yellow color (BGR)

        cv2.imshow('Click Points Full', self.image)


    def returnPoints(self):
        cv2.namedWindow('Click Points Full', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Click Points Full', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.setMouseCallback('Click Points Full', self.mouse_callback)

        while self.click_count < 2:
            key = cv2.waitKey(10)
            if key == 27:  # Exit loop if the 'Esc' key is pressed
                break
        cv2.destroyAllWindows()

        self.Point1 = self.click_points[0]
        self.Point2 = self.click_points[1]

        return (self.Point1,self.Point2)


#---------------------------------------------------------------------------------------
#ByteTracker Tuning
@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 30
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

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
#-----------------------------------------------------------------------------------


def get_frame(src):
    vid = cv2.VideoCapture(src)
    ret, frame = vid.read()
    vid.release()
    global screen_width, screen_height

    # Calculate aspect ratio
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    aspect_ratio = float(frame_width / frame_height)

    # Calculate new dimensions while maintaining aspect ratio
    if screen_width / aspect_ratio <= screen_height:
        new_width = screen_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = screen_height
        new_width = int(new_height * aspect_ratio)

    # Resize the frame
    frame = cv2.resize(frame, (new_width, new_height))
    return frame, new_height, new_width


def process_frame(frame, byte_tracker, line_counters, box_annotator, line_annotators, model):
    global screen_width, screen_height

    # Calculate aspect ratio
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    aspect_ratio = float(frame_width / frame_height)

    # Calculate new dimensions while maintaining aspect ratio
    if screen_width / aspect_ratio <= screen_height:
        new_width = screen_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = screen_height
        new_width = int(new_height * aspect_ratio)

    # Resize the frame
    frame = cv2.resize(frame, (new_width, new_height))

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
        f"#{tracker_id}"
        for _, confidence, class_id, tracker_id
        in detections
    ]

    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    for line_counter, line_annotator in zip(line_counters, line_annotators):
        line_counter.trigger(detections=detections)
        line_annotator.annotate(frame=frame, line_counter=line_counter)

    return frame


def setup_lines(src, lines):
    colors = sv.ColorPalette.default()
    LineCounters = []
    for line in lines:
        point1, point2 = line
        x1, y1 = point1
        x2, y2 = point2
        LINE_START = Point(x1, y1)
        LINE_END = Point(x2, y2)
        line_counter = LineZone(start=LINE_START, end=LINE_END)
        LineCounters.append(line_counter)

    byte_tracker = BYTETracker(BYTETrackerArgs())
    box_annotator = BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.3, text_padding=1)
    line_annotators = [LineZoneAnnotator(color= colors.by_idx(index+1),thickness=2, text_thickness=1, text_scale=0.6, text_padding=1)
                       for index, line_counter in enumerate(LineCounters)]

    frame = get_frame(src)[0]
    for line_counter, line_annotator in zip(LineCounters, line_annotators):
        line_annotator.annotate(frame=frame, line_counter=line_counter)


    return frame, byte_tracker, box_annotator, line_annotators, LineCounters





def cam_count(cam, byte_tracker, Lines, box_annotator, line_annotator):
    model = Model()
    cap = cv2.VideoCapture(cam)
    # Check if the camera is opened
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam)

    print('camera opened')
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        screen_width = get_monitors()[0].width
        screen_height = get_monitors()[0].height

        # Calculate aspect ratio
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        aspect_ratio = frame_width / frame_height

        # Calculate new dimensions while maintaining aspect ratio
        if screen_width / aspect_ratio <= screen_height:
            new_width = screen_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = screen_height
            new_width = int(new_height * aspect_ratio)

        # Resize the frame
        frame = cv2.resize(frame, (new_width, new_height))

        results = model.predict(frame, imgsz=1280)[0]
        detections = sv.Detections.from_yolov8(results)

        # filtering out detections with unwanted classes
        # detections = detections[detections.class_id==0]

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
            f"#{tracker_id}"
            for _, confidence, class_id, tracker_id
            in detections
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
        # updating line counter
        for line_counter in Lines:
            line_counter.trigger(detections=detections)
            line_annotator.annotate(frame=frame, line_counter=line_counter)

        cv2.namedWindow('Output', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Output', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Output', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def vid_count(vid, byte_tracker, line_counters, box_annotator, line_annotators):
    model = Model()
    generator = get_video_frames_generator(vid)
    for frame in generator:
        frame = process_frame(frame, byte_tracker, line_counters, box_annotator, line_annotators, model)

        cv2.namedWindow('Output', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Output', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Output', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()



def CountWalk(source, filepath):
    src = filepath
    if source == 'camera':
        src = '/dev/video0'

    print('source= ', source, '\n', 'file= ', src)
    frame, height, width = get_frame(src)
    lines = []
    for i in range(2):
        lines.append(LINE(frame).returnPoints())

    frame, byte_tracker, box_annotator, line_annotators, LineCounters = setup_lines(src, lines)

    if source == 'camera':
        cam_count(src, byte_tracker, LineCounters, box_annotator, line_annotators)

    elif source == 'file':
        vid_count(src, byte_tracker, LineCounters, box_annotator, line_annotators)


def main(source, filepath):
    # basic startup setup
    ultralytics.checks()
    print("yolox.__version__:", yolox.__version__)
    CountWalk(source=source, filepath=filepath)

if __name__ == '__main__':
    source = sys.argv[1]
    filepath = sys.argv[2] if source == 'file' else ''
    main(source=source,filepath=filepath)
