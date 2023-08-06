import sys
import ultralytics
import yolox
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from dataclasses import dataclass
from supervision import Point
from supervision import VideoInfo
from supervision import get_video_frames_generator
from supervision import Detections, BoxAnnotator
from supervision.detection.line_counter import LineZone, LineZoneAnnotator
from typing import List
import numpy as np
import supervision as sv
import cv2
from utils.LINE import LINE
from utils.setup_files import pre_setup, Model
from screeninfo import get_monitors
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


def cam_count(cam, byte_tracker, line_counter, box_annotator, line_annotator):
    model = Model()
    print('\n\n\n'+cam)
    cap = cv2.VideoCapture(cam)
    # Check if the camera is opened
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        monitors = get_monitors()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if monitors:
            screen = monitors[0]  # Assuming you want information about the primary monitor
            width = screen.width
            height = screen.height
        frame = cv2.resize(frame, (width, height))

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
    cap.release()
    cv2.destroyAllWindows()

def vid_count(vid, byte_tracker, line_counter, box_annotator, line_annotator):
    model = Model()
    generator = get_video_frames_generator(vid)
    # open target video file
    for frame in generator:
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

    vid = cv2.VideoCapture(src)
    ret, frame = vid.read()

    screen_width = get_monitors()[0].width
    screen_height = get_monitors()[0].height

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

    point1, point2 = LINE(frame).returnPoints()
    x1,y1 = point1
    x2,y2 = point2
    LINE_START = Point(x1,y1)
    LINE_END = Point(x2,y2)

    print('VideoInfo: ',end='')
    if source == 'file':
        VideoInfo.from_video_path(src)

    byte_tracker = BYTETracker(BYTETrackerArgs())
    line_counter = LineZone(start=LINE_START, end=LINE_END)
    box_annotator = BoxAnnotator(thickness=2, text_thickness=2, text_scale=1, text_padding=1)
    line_annotator = LineZoneAnnotator(thickness=3, text_thickness=1, text_scale=2, text_padding=1)


    if source == 'camera':
        cam_count(src, byte_tracker, line_counter, box_annotator, line_annotator)

    elif source == 'file':
        vid_count(src, byte_tracker, line_counter, box_annotator, line_annotator)


def main(source, filepath):
    # basic startup setup
    pre_setup()
    ultralytics.checks()
    print("yolox.__version__:", yolox.__version__)
    CountWalk(source=source, filepath=filepath)

if __name__ == '__main__':
    source = sys.argv[1]
    filepath = sys.argv[2] if source == 'file' else ''
    main(source=source,filepath=filepath)
