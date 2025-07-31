import os
from PIL import Image
import glob

def crop_image(input_path, output_path, crop_size=(2464, 2464)):
    """
    Crops the center of an image to the specified size and saves it.
    """
    img = Image.open(input_path)
    width, height = img.size
    crop_width, crop_height = crop_size
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    right = left + crop_width
    bottom = top + crop_height
    cropped_img = img.crop((left, top, right, bottom))
    cropped_img.save(output_path)
    return output_path

def split_image_to_tiles(input_path, output_dir, date_prefix, tile_size=352):
    """
    Splits an image into tiles and saves them with a specific naming convention.
    """
    img = Image.open(input_path)
    width, height = img.size
    os.makedirs(output_dir, exist_ok=True)
    
    tile_id = 1
    for j in range(0, height, tile_size):
        for i in range(0, width, tile_size):
            box = (i, j, i + tile_size, j + tile_size)
            tile = img.crop(box)
            tile_path = os.path.join(output_dir, f"{date_prefix}-{tile_id}.tif")
            tile.save(tile_path)
            tile_id += 1

def process_dataset(base_dir='dataset/mask'):
    """
    Restructures the dataset by cropping and tiling images into date-specific folders.
    """
    band_folders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    
    for band_folder in band_folders:
        image_files = glob.glob(os.path.join(band_folder, '*.tiff'))
        
        for image_path in image_files:
            filename = os.path.basename(image_path)
            date_prefix = os.path.splitext(filename)[0]
            
            # Create a new directory for the date
            date_dir = os.path.join(band_folder, date_prefix)
            os.makedirs(date_dir, exist_ok=True)
            
            # 1. Crop the image
            cropped_image_path = os.path.join(date_dir, 'cropped.tif')
            crop_image(image_path, cropped_image_path)
            
            # 2. Split the cropped image into tiles
            split_image_to_tiles(cropped_image_path, date_dir, date_prefix)
            
            # 3. Clean up the temporary cropped file
            os.remove(cropped_image_path)
            
            # 4. Move the original image to a processed folder (optional)
            processed_dir = os.path.join(band_folder, 'processed_originals')
            os.makedirs(processed_dir, exist_ok=True)
            os.rename(image_path, os.path.join(processed_dir, filename))

if __name__ == "__main__":
    process_dataset()
    print("Dataset processing complete.")
