from ftplib import FTP
import os

server_ip = '144.76.182.206'
username = 'computervision'
password = 'Computervision@253#87'
working_dir = '/ComputerVision/'

def upload_files(folder, delete=False):
    ftp = FTP(host=server_ip, user=username, passwd=password)
    ftp.cwd(working_dir)
    for file in [file for file in os.listdir(folder) if file not in ['.gitkeep', '.gitignore']]:
        with open(folder+file, 'rb') as f:
            # Use the STOR command to upload the file to the FTP server
            ftp.storbinary(f'STOR {folder+file}', f)
        if delete:
            os.remove(folder+file)
    print('\nDirectory Uploaded!\n')
    ftp.quit()

if __name__ == '__main__':
    repo = ''
    upload_files(repo)
