
import os
import shutil

img_root = 'img'
mask_root = 'mask'
bands = ['vh_decibel', 'vh_linear', 'vv_decibel', 'vv_linear']
mask_src = os.path.join(mask_root, 'mask_bw.tif')

 
for band in bands:
    img_band_dir = os.path.join(img_root, band)
    mask_band_dir = os.path.join(mask_root, band)
    if not os.path.exists(mask_band_dir):
        os.makedirs(mask_band_dir)
    for fname in os.listdir(img_band_dir):
        if fname.lower().endswith('.tiff') or fname.lower().endswith('.tif'):
            dst_path = os.path.join(mask_band_dir, fname)
            shutil.copy(mask_src, dst_path)
            print(f'Copied {mask_src} to {dst_path}')
