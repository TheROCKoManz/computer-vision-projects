from ftplib import FTP
import os

server_ip = '144.76.182.206'
username = 'computervision'
password = 'Computervision@253#87'
working_dir = '/ComputerVision/'

def download_folder_content(folder, targets=[]):
    # Connect to the FTP server
    ftp = FTP(host=server_ip, user=username, passwd=password)
    ftp.cwd(working_dir+folder)

    # Get the list of files in the remote folder
    file_list = ftp.nlst()

    # Download each file from the remote folder to the local folder
    for file_name in [file for file in file_list if file.lower() not in ['.gitkeep', '.gitignore']]:
        if file_name.lower().split('_')[0] in targets:
            print('\nDownloading video of '+file_name.lower().split('_')[0]+'...')
            local_file_path = f'{folder}/{file_name}'
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

if __name__ == '__main__':
    download_videos()

