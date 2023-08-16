import sys
import ultralytics
from supervision import VideoInfo
from supervision import get_video_frames_generator
import numpy as np
import supervision as sv
import cv2
import os
from utils.tools import Polygon, Model
from screeninfo import get_monitors

# ---------------------------------------------------------------------------------------

def process_camera(cam, zones, zone_annotators, box_annotators):
    model = Model()
    cap = cv2.VideoCapture(cam)
    # Check if the camera is opened
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam)

    print('Camera opened')
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        end = process_frame(frame, zones, zone_annotators, box_annotators, model)
        if end:
            break
    cv2.destroyAllWindows()


def process_video(vid, zones, zone_annotators, box_annotators):
    model = Model()
    generator = get_video_frames_generator(vid)

    for frame in generator:
        end = process_frame(frame, zones, zone_annotators, box_annotators, model)
        if end:
            break
    cv2.destroyAllWindows()

def process_frame(frame, zones, zone_annotators, box_annotators, model):
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
    detections = detections[detections.class_id == 0]

    for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
        mask = zone.trigger(detections=detections)
        detections_filtered = detections[mask]
        frame = box_annotator.annotate(scene=frame, detections=detections_filtered)
        frame = zone_annotator.annotate(scene=frame)

    cv2.namedWindow('Output', cv2.WND_PROP_FULLSCREEN)
    cv2.imshow('Output', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return True


def setup_zones(src):
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

    colors = sv.ColorPalette.default()
    # Resize the frame
    frame = cv2.resize(frame, (new_width, new_height))
    cv2.namedWindow('Sample Frame press ESC to exit', cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty('Sample Frame press ESC to exit', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Sample Frame press ESC to exit', frame)

    while True:
        # Wait for a key event (0 delay means wait indefinitely)
        key = cv2.waitKey(0) & 0xFF

        # Check if the 'Esc' key is pressed
        if key == 27:
            break

        # Close all OpenCV windows
    cv2.destroyAllWindows()  # Provide your own frame

    colors = sv.ColorPalette.default()
    polygons = []

    No_of_Zones = int(input('\nEnter number of Zones of Interest: '))

    if No_of_Zones == 0:
        default_zone = np.array([(0, 0), (new_width, 0), (new_width, new_height), (0, new_height)])
        polygons = [default_zone]

    for i in range(No_of_Zones):
        zone = Polygon(frame=frame, colour=colors.by_idx(i)).returnPoints()
        zone = np.array(zone, np.int32)
        polygons.append(zone)

    zones = [sv.PolygonZone(polygon=polygon, frame_resolution_wh=(frame.shape[1], frame.shape[0]))
             for polygon in polygons]

    zone_annotators = [sv.PolygonZoneAnnotator(zone=zone,
                                               color=colors.by_idx(index),
                                               thickness=2,
                                               text_thickness=1,
                                               text_scale=2,
                                               text_padding=1)
                       for index, zone in enumerate(zones)]

    box_annotators = [sv.BoxAnnotator(color=colors.by_idx(index),
                                      thickness=1,
                                      text_thickness=1,
                                      text_scale=0.35,
                                      text_padding=1)
                      for index in range(len(polygons))]


    for zone_annotator in zone_annotators:
        frame = zone_annotator.annotate(scene=frame)

    cv2.namedWindow('Selected Zones...press ESC to exit', cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty('Sample Frame press ESC to exit', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Selected Zones...press ESC to exit', frame)

    while True:
        # Wait for a key event (0 delay means wait indefinitely)
        key = cv2.waitKey(0) & 0xFF

        # Check if the 'Esc' key is pressed
        if key == 27:
            break

        # Close all OpenCV windows
    cv2.destroyAllWindows()

    return zones, zone_annotators, box_annotators


def CountinZone(source, filepath=''):
    ultralytics.checks()
    src = filepath if source == 'file' else '/dev/video0'

    print('Source:', source, '\nFile:', src)

    zones, zone_annotators, box_annotators = setup_zones(src)

    if source == 'camera':
        process_camera(src, zones, zone_annotators, box_annotators)
    elif source == 'file':
        process_video(src, zones, zone_annotators, box_annotators)



if __name__ == '__main__':
    source = sys.argv[1]
    filepath = sys.argv[2] if source == 'file' else ''
    CountinZone(source=source, filepath=filepath)
