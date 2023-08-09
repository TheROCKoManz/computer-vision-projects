from ftplib import FTP, all_errors
import os

server_ip = '144.76.182.206'
username = 'computervision'
password = 'Computervision@253#87'
working_dir = '/ComputerVision/Data/Facial_Recog/Raw_DataStore/FacialRecog_TargetVideo/'

def upload_filex(file_path, delete=False):
    try:
        # Establish FTP connection
        ftp = FTP(host=server_ip, user=username, passwd=password)
        ftp.cwd(working_dir)

        # Extract the filename from the full path
        file_name = os.path.basename(file_path)

        # Upload the file
        with open(file_path, 'rb') as f:
            ftp.storbinary(f'STOR {file_name}', f)

        if delete:
            os.remove(file_path)

        print(f"File {file_name} uploaded successfully.")

    except all_errors as e:
        print("FTP error:", e)



if __name__ == '__main__':
    file = ''
    upload_filex(file)
