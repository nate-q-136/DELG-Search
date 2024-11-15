from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os


def upload_to_drive(filename):
    # Authenticate and create the PyDrive client
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication
    drive = GoogleDrive(gauth)

    # Create and upload the file to Google Drive
    file_drive = drive.CreateFile({'title': os.path.basename(filename)})
    file_drive.SetContentFile(filename)
    file_drive.Upload()
    print(f"File {filename} uploaded to Google Drive.")