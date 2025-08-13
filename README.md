# Satellite Image Processing for Land Cover Classification

This project outlines the workflow for downloading and preprocessing Sentinel-1 satellite imagery to generate training data for a machine learning model.                                                      

## 🛰️ 1. Data Acquisition

Follow these steps to download the required Sentinel-1 images from the [Copernicus Open Access Hub](https://scihub.copernicus.eu/).

* **Navigate to Data Collections and select the following:**
    * **Collection:** Sentinel-1
    * **Acquisition Mode:** IW - Interferometric Wide Swath (10m x 10m)
    * **Polarization:** `VV` + `VH`

* **Define Area of Interest (AOI):**
    * Upload the provided `area_of_interest.zip` file to define the geographical boundaries for the image search.

* **Select Acquisition Dates:**
    * Set the time interval to include the following dates:
        * `2015-03-29`
        * `2015-04-22`
        * `2015-05-16`
        * `2015-07-27`
        * `2015-09-13`
        * `2015-10-07`
        * `2015-11-24`
        * `2015-12-18`
        * `2016-01-11`
        * `2016-02-04`
        * `2016-02-28`
        * `2016-03-23`
        * `2016-04-16`
        * `2016-05-10`
        * `2016-06-03`
        * `2016-06-27`
        * `2016-07-21`
        * `2016-08-14`

* **Download Image Settings:**
    * In the download options, select the **Analytical** tab and configure the following:
        * **Image Format:** TIFF (16-bit)
        * **Image Resolution:** High
        * **Coordinate System:** WGS 84 (EPSG:4326)
        * **Bands:** Select both `VH - decibel gamma()` and `VV - decibel gamma()`

***

## 🖼️ 2. Preprocessing

Once all images are downloaded, preprocess them to create standardized training tiles.

1.  **Organize Files:**
    * Extract all downloaded `.tif` files into a single folder named `downloaded_copernicus_image`.

2.  **Reconstruct Directories:**
    * Run the script to organize the raw images into a structured directory format.
    ```bash
    python data_preprocessing/directory_reconstruction.py
    ```

3.  **Verify Image Resolution:**
    * Run the checker script to ensure all images have a consistent resolution of **2483 x 2500** pixels.
    ```bash
    python image_resolution_checker.py
    ```

4.  **Generate Training Tiles:**
    * Run the resizing and cropping script to generate smaller image tiles from the full-resolution images. These tiles will be used for model training.
    ```bash
    python image_resizing_cropping.py
    ```

***

## 🎭 3. Mask Generation

The final step is to generate the corresponding masks for the training tiles.

1.  **Generate Grayscale Masks:**
    * Run the script to create the single-channel grayscale masks required for training.
    ```bash
    python mask_generation.py
    ```

2.  **(Optional) Generate RGB Masks:**
    * If you need a visual representation of the masks for review, run this script to generate a 3-channel RGB version.
    ```bash
    python mask_rgb_generation.py
    ```
