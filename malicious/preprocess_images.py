
import os
import shutil

# The source directory containing the .tiff files
source_dir = 'img'

# The names of the destination folders
categories = ['vv_decibel', 'vh_decibel', 'vv_linear', 'vh_linear']

# Create the destination folders if they don't exist
for category in categories:
    if not os.path.exists(category):
        os.makedirs(category)

# Get a list of all files in the source directory
try:
    file_list = os.listdir(source_dir)
except FileNotFoundError:
    print(f"Error: The directory '{source_dir}' was not found.")
    exit()

# Iterate over each file
for filename in file_list:
    # Check if the file is a .tiff file
    if filename.endswith('.tiff'):
        # Determine the category based on the filename
        dest_folder = None
        if '_VV_-_' in filename and '_decibel_' in filename:
            dest_folder = 'vv_decibel'
        elif '_VH_-_' in filename and '_decibel_' in filename:
            dest_folder = 'vh_decibel'
        elif '_VV_-_' in filename and '_linear_' in filename:
            dest_folder = 'vv_linear'
        elif '_VH_-_' in filename and '_linear_' in filename:
            dest_folder = 'vh_linear'

        if dest_folder:
            # Extract the date from the beginning of the filename
            date_part = filename[:10]
            
            # Create the new filename
            new_filename = f"{date_part}.tiff"
            
            # Construct the full source and destination paths
            source_path = os.path.join(source_dir, filename)
            destination_path = os.path.join(dest_folder, new_filename)
            
            # Move and rename the file
            try:
                shutil.move(source_path, destination_path)
                print(f"Moved and renamed '{filename}' to '{destination_path}'")
            except Exception as e:
                print(f"Error moving file {filename}: {e}")
        else:
            print(f"Could not categorize file: {filename}")

print("\nProcessing complete.")
