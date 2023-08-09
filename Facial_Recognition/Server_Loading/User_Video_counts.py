import ftplib

def count_user_videos(user_id):
    # Replace these with your FTP server details
    server_ip = '144.76.182.206'
    username = 'computervision'
    password = 'Computervision@253#87'
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

# Example usage:
user_id = 'BillGates' # Replace with the actual User_ID
video_count = count_user_videos(user_id)

if video_count >= 0:
    print(f"User {user_id} has {video_count} videos.")
else:
    print("Error occurred while counting user videos.")
