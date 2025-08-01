import os
import glob
import numpy as np
import rasterio
from skimage.feature import graycomatrix, graycoprops
from skimage.util import view_as_windows
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm
import warnings

# Suppress RuntimeWarning from mean of empty slice
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

def scale_to_uint8(image_data: np.ndarray) -> np.ndarray:
    """Scales numpy array to uint8 (0-255)."""
    image_data = np.nan_to_num(image_data)
    min_val, max_val = np.min(image_data), np.max(image_data)
    if max_val == min_val:
        return np.zeros(image_data.shape, dtype=np.uint8)
    scaled_data = 255.0 * (image_data - min_val) / (max_val - min_val)
    return scaled_data.astype(np.uint8)

def calculate_windowed_glcm_features(
    image_uint8: np.ndarray, 
    window_size: int = 7, 
    properties: tuple = ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM')
) -> np.ndarray:
    """
    Calculates GLCM features for an image using a sliding window.
    
    NOTE: This is a computationally expensive operation.
    """
    ws = window_size
    h, w = image_uint8.shape
    num_features = len(properties)
    
    # Pad the image to handle borders
    pad = ws // 2
    padded_image = np.pad(image_uint8, pad, mode='reflect')
    
    # Create windows
    windows = view_as_windows(padded_image, (ws, ws))
    
    # Initialize feature maps
    feature_maps = np.zeros((h, w, num_features), dtype=np.float32)
    
    # GLCM parameters
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    # Iterate over each window
    for y in range(h):
        for x in range(w):
            window = windows[y, x]
            
            # Skip empty windows if they occur
            if window.max() == window.min():
                feature_maps[y, x, :] = 0
                continue

            glcm = graycomatrix(window, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
            
            feature_vector = [np.mean(graycoprops(glcm, prop)) for prop in properties]
            feature_maps[y, x, :] = feature_vector
            
    return feature_maps

def process_images(vv_path: str, vh_path: str, out_path: str):
    """
    Reads VV and VH images, calculates GLCM, performs PCA, and saves the result.
    """
    # 1. Read images and get metadata
    with rasterio.open(vv_path) as src:
        profile = src.profile
        vv_data = src.read(1)
    with rasterio.open(vh_path) as src:
        vh_data = src.read(1)

    # 2. Scale to uint8 for GLCM
    vv_uint8 = scale_to_uint8(vv_data)
    vh_uint8 = scale_to_uint8(vh_data)

    # 3. Calculate GLCM features for both polarizations
    # This is the slowest part of the script.
    glcm_props = ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM')
    features_vv = calculate_windowed_glcm_features(vv_uint8, properties=glcm_props)
    features_vh = calculate_windowed_glcm_features(vh_uint8, properties=glcm_props)

    # 4. Combine features and reshape for PCA
    h, w, _ = features_vv.shape
    combined_features = np.concatenate((features_vv, features_vh), axis=2)
    num_total_features = len(glcm_props) * 2
    reshaped_features = combined_features.reshape((h * w, num_total_features))

    # 5. Scale features and apply PCA
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(reshaped_features)
    
    pca = PCA(n_components=1)
    principal_component = pca.fit_transform(scaled_features)
    
    # Reshape PC1 back to image dimensions
    pc1_image = principal_component.reshape((h, w))

    # 6. Save the output GeoTIFF
    profile.update(dtype=rasterio.float32, count=1, compress='lzw')
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(pc1_image.astype(rasterio.float32), 1)

def main():
    """
    Main function to orchestrate the processing of the dataset.
    """
    train_base_path = r'dataset_newest/train'
    
    date_folders = [d for d in os.listdir(train_base_path) if os.path.isdir(os.path.join(train_base_path, d))]

    print(f"Found {len(date_folders)} date folders to process.")

    for date_folder in tqdm(date_folders, desc="Overall Progress"):
        img_folder = os.path.join(train_base_path, date_folder, 'img')
        vv_folder = os.path.join(img_folder, 'vv_decibel')
        vh_folder = os.path.join(img_folder, 'vh_decibel')
        glcm_folder = os.path.join(img_folder, 'glcm')

        if not os.path.isdir(vv_folder) or not os.path.isdir(vh_folder):
            continue

        os.makedirs(glcm_folder, exist_ok=True)

        vv_images = glob.glob(os.path.join(vv_folder, '*.tif'))
        
        for vv_image_path in tqdm(vv_images, desc=f"Date {date_folder}", leave=False):
            base_name = os.path.basename(vv_image_path)
            vh_image_path = os.path.join(vh_folder, base_name)
            output_path = os.path.join(glcm_folder, base_name)

            if not os.path.exists(vh_image_path):
                print(f"Warning: No VH image for {vv_image_path}. Skipping.")
                continue
            
            if os.path.exists(output_path):
                # print(f"Output exists for {base_name}. Skipping.")
                continue

            try:
                process_images(vv_image_path, vh_image_path, output_path)
            except Exception as e:
                print(f"\nCould not process {base_name}. Error: {e}")

    print("\nProcessing complete.")

if __name__ == '__main__':
    main()