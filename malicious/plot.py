import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Paths to each folder for each image type
folders = {
    'vv_decibel': 'dataset/img/vv_decibel',
    'vv_linear': 'dataset/img/vv_linear',
    'vh_decibel': 'dataset/img/vh_decibel',
    'vh_linear': 'dataset/img/vh_linear'
}

# Get a sorted list of unique dates based on filenames in any one folder
def get_dates(folder):
    file_names = os.listdir(folder)
    dates = [os.path.splitext(f)[0] for f in file_names]
    return sorted(dates)

# Get intersection of dates present in all folders
dates_per_folder = [set(get_dates(folder)) for folder in folders.values()]
all_dates = sorted(list(set.intersection(*dates_per_folder)))

# Create plot
n_rows = len(folders)
n_cols = len(all_dates)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
fig.subplots_adjust(hspace=0.1, wspace=0.05)

# Load and plot images
row_labels = list(folders.keys())
for row_idx, folder_key in enumerate(row_labels):
    folder = folders[folder_key]
    for col_idx, date in enumerate(all_dates):
        img_path = os.path.join(folder, f"{date}.png")  # or .jpg, adjust as needed
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[row_idx, col_idx].imshow(img)
        axes[row_idx, col_idx].axis('off')
        # Optionally add date labels at the top
        if row_idx == 0:
            axes[row_idx, col_idx].set_title(date, fontsize=8)
    # Optionally add row labels on the left
    axes[row_idx, 0].set_ylabel(folder_key, rotation=0, labelpad=40, fontsize=10, va="center")

plt.tight_layout()
plt.show()