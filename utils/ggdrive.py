from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os


def upload_to_drive(filename, folder_id=None):
    # Authenticate and create the PyDrive client
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()  # Creates local webserver and auto handles authentication
    drive = GoogleDrive(gauth)

    # Create the file to upload
    file_drive = drive.CreateFile({
        'title': os.path.basename(filename),
        'parents': [{'id': folder_id}] if folder_id else None  # Add folder_id if provided
    })
    file_drive.SetContentFile(filename)
    file_drive.Upload()
    print(f"File {filename} uploaded to Google Drive in folder ID {folder_id if folder_id else 'root'}.")