import os
from typing import List

def get_folder_names(directory_path: str) -> List[str]:
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory {directory_path} does not exist.")
    
    folder_names = [
        entry for entry in os.listdir(directory_path) 
        if os.path.isdir(os.path.join(directory_path, entry))
    ]
    return folder_names

if __name__ == "__main__":
    # Define the path to the directory
    images_dir = "artifacts/data_ingestion/Images"
    
    try:
        # Get folder names
        names_in_images = get_folder_names(images_dir)
        print("Folder names in 'Images':", names_in_images)
    except Exception as e:
        print(f"An error occurred: {e}")
