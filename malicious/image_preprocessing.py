from PIL import Image
import os

def crop_image(input_path, output_path, crop_size=(2464, 2464)):
    """
    Crop the center of the image to the specified crop_size and save it.
    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the cropped image.
        crop_size (tuple): (width, height) of the crop.
    Returns:
        None
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

def split_image_to_tiles(input_path, output_dir, tile_size=352):
    """
    Split a 2464x2464 image into 352x352 tiles and save them to output_dir.
    Args:
        input_path (str): Path to the input image.
        output_dir (str): Directory to save the tiles.
        tile_size (int): Size of each tile (default 352).
    Returns:
        None
    """
    import os
    img = Image.open(input_path)
    width, height = img.size
    assert width == 2464 and height == 2464, "Image size must be 2464x2464"
    os.makedirs(output_dir, exist_ok=True)
    tile_num = 0
    for i in range(0, width, tile_size):
        for j in range(0, height, tile_size):
            box = (i, j, i + tile_size, j + tile_size)
            tile = img.crop(box)
            tile_num += 1
            tile_path = os.path.join(output_dir, f"{tile_num}.png")
            tile.save(tile_path)
    print(f"Saved {tile_num} tiles to {output_dir}")


crop_image('mask_rgb.png', 'mask_rgb_crop.png', crop_size=(2464, 2464))
split_image_to_tiles('mask_rgb_crop.png', 'test/', tile_size=352)
