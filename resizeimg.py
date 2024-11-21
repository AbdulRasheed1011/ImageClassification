"""
Use this code for resize images using OpenCV.
"""
import cv2
import os

def load_and_resize_image(file_path, image_size):
    """
    Load an image from a file path and resize it to the given size in RGB mode.
    
    Args:
        file_path (str): Path to the image file.
        image_size (int): The desired width and height for resizing the image.
    
    Returns:
        numpy.ndarray: The resized RGB image, or None if the image couldn't be read.
    """
    img_array = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if img_array is None:
        print(f"Warning: {file_path} could not be read and will be skipped.")
        return None
    resized_image = cv2.resize(img_array, (image_size, image_size))
    return resized_image

def save_image(image, output_path):
    """
    Save an image to a specified output path.
    
    Args:
        image (numpy.ndarray): The image to save.
        output_path (str): The path where the image will be saved.
    """
    cv2.imwrite(output_path, image)
    print(f"Saved resized image to {output_path}")

def process_images(input_folder, output_folder, image_size):
    """
    Process all images in the input folder by resizing and saving them to the output folder.
    
    Args:
        input_folder (str): Path to the folder containing the input images.
        output_folder (str): Path to the folder where resized images will be saved.
        image_size (int): The desired width and height for resizing the images.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        resized_image = load_and_resize_image(file_path, image_size)
        
        if resized_image is not None:
            output_path = os.path.join(output_folder, file_name)
            save_image(resized_image, output_path)

# Example usage
if __name__ == "__main__":
    input_folder = "path/to/your/image/folder"
    output_folder = "path/to/output/folder2"
    image_size = 70
    process_images(input_folder, output_folder, image_size)
