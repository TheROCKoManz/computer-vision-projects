from ftplib import FTP
import os

server_ip = 'X.X.X.X'
username = 'computervision'
password = 'XXXXXXXX'
working_dir = '/ComputerVision/'

def download_folder_content(folder, targets=[]):
    ftp = FTP(host=server_ip, user=username, passwd=password)
    ftp.cwd(working_dir+folder)

    file_list = ftp.nlst()

    # Download each file from the remote folder to the local folder
    for file_name in [file for file in file_list if file.lower() not in ['.gitkeep', '.gitignore']]:
        if file_name.lower().split('_')[0] in targets:
            print('\nDownloading video of '+file_name.lower().split('_')[0]+'...')
            local_file_path = f'{folder}{file_name}'
            if not os.path.exists(local_file_path):
                with open(local_file_path, 'wb') as local_file:
                    ftp.retrbinary(f'RETR {file_name}', local_file.write)
            print('Downloaded video of ' + file_name.lower().split('_')[0] + '...\n')

    # Close the FTP connection
    ftp.quit()

def download_videos(targets):
    print("\nDownloading Training Videos...\n")
    download_folder_content('Data/Facial_Recog/Raw_DataStore/FacialRecog_TargetVideo/', targets)
    print(f'Raw videos downloaded for training.')


