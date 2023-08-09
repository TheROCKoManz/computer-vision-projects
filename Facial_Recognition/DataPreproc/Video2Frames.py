import cv2 as cv
import supervision as sv

# USAGE----------------------------------------------------------------
# VideoCapture(): INPUTS Params
#           name ---------> frame-prefix
#           input_path ---> full path of video feed
#           save_path ----> full path to save frames
#           time_limit ---> limit time for clip
#----------------------------------------------------------------------

def Video2Frames(name, input_path, save_path, time_limit, skip_rate=1):
    # ---> Frame-name suffix
    counter = 100001
    target_name = name
    videoinfo = sv.VideoInfo.from_video_path(input_path)
    vid_resol = (videoinfo.width,videoinfo.height)
    fps = videoinfo.fps
    video_generator = sv.get_video_frames_generator(input_path)
    for frame in video_generator:
        if time_limit >= 0:
            frame = cv.resize(frame, vid_resol)
            # saving frames
            if counter % skip_rate == 0:
                print(save_path + "/" + target_name + str(counter) + r".jpg")
                cv.imwrite(save_path + "/" + target_name +'_'+ str(counter) + r".jpg", frame)
            counter += 1  # --->increment suffix counter

            # calculate elapsed time of video to limit time
            if counter % int(fps) == 0:
                time_limit -= 1
        else:
            break


