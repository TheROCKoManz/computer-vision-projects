import ftplib

def count_user_videos(user_id):
    # Replace these with your FTP server details
    server_ip = 'x.x.x.x'
    username = 'computervision'
    password = 'xxxxxxxx'
    working_dir = '/ComputerVision/Data/Facial_Recog/Raw_DataStore/FacialRecog_TargetVideo'

    try:
        with ftplib.FTP(server_ip) as ftp:
            ftp.login(username, password)
            ftp.cwd(working_dir)
            user_video_count = 0

            # Iterate through files in the directory
            for filename in ftp.nlst():
                parts = filename.split('_')
                if len(parts) == 2 and parts[0] == str(user_id):
                    user_video_count += 1

            return user_video_count

    except ftplib.all_errors as e:
        print("FTP error:", e)
        return -1  # Return a negative value to indicate an error

