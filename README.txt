1. Download Sentinel-1 Images from Copernicus Browser
- In Data Collections,
	- Choose Sentinel-1.
	- Choose IW - Interferometric Wide Swath 10m x 10m Acquisition mode
	- Choose Polarization: VV + VH
- Choose Area of Interest by uploading area_of_interest.zip to the website
- Select these dates:
	- 2015-03-29
	- 2015-04-22
	- 2015-05-16
	- 2015-07-27
	- 2015-09-13
	- 2015-10-07
	- 2015-11-24
	- 2015-12-18
	- 2016-01-11
	- 2016-02-04
	- 2016-02-28
	- 2016-03-23
	- 2016-04-16
	- 2016-05-10
	- 2016-06-03
	- 2016-06-27
	- 2016-07-21
	- 2016-08-14

- Download image, select Analytical Tab
	- Choose Image Format: TIFF (16-bit)
	- Choose Image Resolution: High
	- Choose Coordinate System: WSG 84 (EPSG:4326)
	- Choose band: VH - decibel gamma() and VV - decibel gamma()

2. Directory reconstruction and Preprocessing images
- Extracted all the downloaded .tif file to Folder downloaded_copernicus_image
- Run data_preprocessing/directory_reconstruction.py
- Run image_resolution_checker.py to make sure every images' resolution is 2483 x 2500
- Run image_resizing_cropping.py to generate tiles. These tiles are used for training

3. Mask Generation
- Run mask_generation.py. The grayscale mask with be used to train the model.
- If you want to see the RGB version of mask file, run mask_rgb_generation.py