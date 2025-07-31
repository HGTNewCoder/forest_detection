
import os
from PIL import Image

# The directories to check
directories = ['vv_decibel', 'vh_decibel', 'vv_linear', 'vh_linear']

# The expected resolution
expected_width = 2483
expected_height = 2500

# A list to store the names of files with incorrect resolutions
incorrect_resolution_files = []

# Iterate over each directory
for directory in directories:
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Directory '{directory}' not found, skipping.")
        continue

    # Get a list of all files in the directory
    try:
        file_list = os.listdir(directory)
    except FileNotFoundError:
        print(f"Error: The directory '{directory}' was not found.")
        continue

    # Iterate over each file in the directory
    for filename in file_list:
        # Check if the file is a .tiff file
        if filename.endswith('.tiff'):
            file_path = os.path.join(directory, filename)
            try:
                # Open the image and get its dimensions
                with Image.open(file_path) as img:
                    width, height = img.size
                    # Check if the resolution matches the expected resolution
                    if width != expected_width or height != expected_height:
                        incorrect_resolution_files.append(file_path)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

# Print the results
if incorrect_resolution_files:
    print("Files with incorrect resolution:")
    for file_path in incorrect_resolution_files:
        print(file_path)
else:
    print("All TIFF files have the correct resolution of 2483x2500.")
