# download_file.py
import gdown

def download_file(file_id: str, output_path: str) -> None:
    """
    Downloads a file from Google Drive using its file ID.

    Args:
        file_id (str): The Google Drive file ID.
        output_path (str): The path where the file should be saved.
    
    Returns:
        None
    """
    try:
        # Construct the Google Drive URL from the file ID
        url = f'https://drive.google.com/uc?id={file_id}'
        
        # Download the file from the Google Drive link
        gdown.download(url, output_path, quiet=False)
        print(f"File downloaded successfully to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
