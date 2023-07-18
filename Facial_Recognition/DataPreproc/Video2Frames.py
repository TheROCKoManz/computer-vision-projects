import cv2 as cv

# USAGE----------------------------------------------------------------
# VideoCapture(): INPUTS Params
#           name ---------> frame-prefix
#           input_path ---> full path of video feed
#           save_path ----> full path to save frames
#           time_limit ---> limit time for clip
#----------------------------------------------------------------------


class Video2Frames:
    ## Constructor for performing basic configurations
    def __init__(self, name, input_path, save_path, time_limit, skip_rate=1):
        self.cap_obj = cv.VideoCapture(input_path)                      # capturing Video-file as object
        self.height = int(self.cap_obj.get(cv.CAP_PROP_FRAME_HEIGHT))   # height of video
        self.width = int(self.cap_obj.get(cv.CAP_PROP_FRAME_WIDTH))     # width of video
        self.vid_resol = (self.width, self.height)                      # video resolution
        self.fps = self.cap_obj.get(cv.CAP_PROP_FPS)                    # video FPS
        self.obj_name = name                                            # prefix of generated frames

        # Handling feed source error ----> Faulty Path or Invalid Device
        if not self.cap_obj.isOpened():
            print("Recheck the input video path/hardware.")

        self.save_path = save_path          # frames destination
        self.skip_rate=skip_rate            # frames skip rate
        self.time_limit = time_limit        # video time limit

        self.get_frames()       #---> get frames from video feed


    ## function to display the video feed
    def display(self):
        while self.cap_obj.isOpened():  #---> to check if video object is open
            con, frame = self.cap_obj.read()    #---> to read frames from open video object
            # con   ---> frame validity
            # frame ---> actual frame

            frame = cv.resize(frame, self.vid_resol)  #--->resize frames to required resolution
            frame = cv.putText(frame,  str(self.fps), (50, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            if con:  # display valid frame, else break
                cv.imshow("Feed", frame)
                if cv.waitKey(1) == ord("q"):  #press 'Q' to stop display feed
                    break
            else:
                break



    ## function to save all frames at a destination directory
    ## store all frames within provided time limit
    def get_frames(self):
        counter = 100000   #---> Frame-name suffix

        # loop till time limit
        while self.cap_obj.isOpened() and self.time_limit >= 0:
            con, frame = self.cap_obj.read()
            if con:
                frame = cv.resize(frame, self.vid_resol)

                # saving frames
                if counter % self.skip_rate == 0:
                    cv.imwrite(self.save_path + r"\\" + self.obj_name + str(counter) + r".jpg", frame)
                counter += 1  #--->increment suffix counter

                # calculate elapsed time of video to limit time
                if counter % int(self.fps) == 0:
                    self.time_limit -= 1
            else:
                break

    ## function to return basic video info
    def get_info(self):
        info = f'Name: {self.obj_name} \nResolution: {self.vid_resol} \nFPS: {self.fps}'
        return info
