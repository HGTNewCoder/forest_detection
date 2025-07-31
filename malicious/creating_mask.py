import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

forest_treatments = gpd.read_file('groundtruth/forest_treatments.shp')
illegal_logging = forest_treatments.loc[forest_treatments['Treatment'] == 'illegal logging']
illegal_concessionary = forest_treatments.loc[forest_treatments['Treatment'] == 'concessionary plus illegal logging']
research_area = gpd.read_file('research_area_convex_square/research_area_convex_square/shifted_square.shp')
illegal_total = pd.concat([illegal_logging, illegal_concessionary])

# --- Create semantic segmentation mask ---
# Define output raster properties
W_PIXEL_SIZE = 2500
H_PIXEL_SIZE = 2483

out_shape = (W_PIXEL_SIZE, H_PIXEL_SIZE)  # You can adjust this resolution as needed

# Get bounds and transform from the research_area (shifted_square)
minx, miny, maxx, maxy = research_area.total_bounds
transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, out_shape[1], out_shape[0])


# Prepare shapes for rasterization (black & white: 1 for legal area, 0 for everything else)
shapes = []
# 1 for square (legal area)
for geom in research_area.geometry:
    shapes.append((geom, 1))
# 0 for illegal areas (overwrite square)
for geom in illegal_total.geometry:
    shapes.append((geom, 0))

# Rasterize: illegal areas (0) will overwrite square (1)
mask = rasterize(
    shapes,
    out_shape=out_shape,
    transform=transform,
    fill=0,  # outside square is 0
    dtype='uint8',
    all_touched=True
)

# Save mask as GeoTIFF

# Save black & white mask as GeoTIFF
with rasterio.open(
    'mask_bw.tif', 'w',
    driver='GTiff',
    height=out_shape[0],
    width=out_shape[1],
    count=1,
    dtype=mask.dtype,
    crs=research_area.crs,
    transform=transform
) as dst:
    dst.write(mask, 1)

print('Mask saved as mask_bw.tif')

# Save as PNG for quick visualization
from PIL import Image
Image.fromarray(mask * 255).save('mask_bw.png')
print('Black & white mask saved as mask_bw.png')

# Print unique values for verification
print(np.unique(mask, return_counts=True))

