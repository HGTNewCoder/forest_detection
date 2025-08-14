# Detecting illegal logging on Sentinel-1 SAR with DeepLabV3-UNet

This repository implements a deep-learning pipeline to detect illegal logging in the Mawas Conservation Area (Central Kalimantan, Indonesia) using Sentinel‑1 SAR imagery. It combines a DeepLabV3 encoder with an EfficientNet‑B4 backbone and a UNet-style decoder enhanced by SCSE attention. The model is trained for binary semantic segmentation of illegal logging vs. background on tiled VH, VV, and VV/VH input bands. Results in the paper report F1 ≈ 0.656 and IoU ≈ 0.488, demonstrating strong potential for operational monitoring with further refinement.

---

## Quick start

> If your local repo structure or filenames differ, adapt the paths below. This README reflects the paper and standard deep-learning project patterns.

```bash
# 1) Clone
git clone https://github.com/HGTNewCoder/forest_detection.git
cd forest_detection

# 2) Create environment
conda create -n forest-detect python=3.10 -y
conda activate forest-detect

# 3) Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # or CPU index-url
pip install segmentation-models-pytorch efficientnet-pytorch
pip install albumentations opencv-python numpy scipy
pip install rasterio shapely geopandas pyproj
pip install scikit-image scikit-learn tqdm pyyaml
pip install matplotlib seaborn

# 4) Organize data
# data/raw/  (SAR VV, VH, ratio)
# data/labels/ (binary masks)

# 5) Preprocess
python scripts/preprocess.py \
  --input_dir data/raw \
  --label_dir data/labels \
  --output_dir data/tiles \
  --crop_size 2464 \
  --tile_size 382 \
  --bands VV,VH,RATIO \
  --val_split 0.2 \
  --filter_tiles 15

# 6) Train
python train.py \
  --data_dir data/tiles \
  --epochs 450 \
  --batch_size 8 \
  --lr 3e-4 \
  --max_lr 1e-3 \
  --weight_decay 1e-5 \
  --loss focal_tversky \
  --focal_alpha 0.6 --focal_gamma 2.0 \
  --tversky_alpha 0.4 --tversky_beta 0.6 \
  --amp \
  --save_dir outputs/run_01

# 7) Evaluate
python eval.py \
  --model_ckpt outputs/run_01/best.ckpt \
  --data_dir data/tiles/val \
  --threshold 0.5

# 8) Inference on a new scene
python infer.py \
  --scene_path data/raw/S1_scene.tif \
  --model_ckpt outputs/run_01/best.ckpt \
  --bands VV,VH,RATIO \
  --tile_size 382 \
  --overlap 32 \
  --threshold 0.5 \
  --out_raster outputs/run_01/pred_illegal_logging.tif
```

---

## Project structure

```
forest_detection/
├─ README.md
├─ train.py
├─ eval.py
├─ infer.py
├─ models/
│  ├─ deeplab_unet_scse.py
│  ├─ backbones.py
│  ├─ losses.py
│  └─ utils.py
├─ data/
│  ├─ raw/
│  ├─ labels/
│  └─ tiles/
├─ scripts/
│  ├─ preprocess.py
│  └─ visualize.py
├─ cfg/
│  └─ default.yaml
├─ outputs/
│  └─ run_01/
├─ requirements.txt
└─ LICENSE
```

---

## Data and preprocessing

- **Area:** ~750 km² within Mawas Conservation Area, Central Kalimantan, Indonesia  
- **Sensor/mode:** Sentinel‑1 IW swath; 250 km swath; 5m × 20m resolution  
- **Polarizations:** VV, VH, and VV/VH ratio  
- **Temporal span:** Mar 2015 – Dec 2016 (23 scenes)  
- **Preprocessing:** Crop to 2464×2464, tile to 382×382, filter tiles to balance classes, normalize bands, augment (rotation, flips, brightness/contrast, Gaussian noise, coarse dropout)

---

## Model architecture

- **Encoder:** DeepLabV3 with EfficientNet‑B4 backbone (ImageNet pretrained)  
- **ASPP:** Multi-dilation aggregation  
- **Decoder:** UNet style with skip connections  
- **Attention:** SCSE blocks  
- **Loss:** 0.6 × Focal loss (γ=2.0, α=0.6) + 0.4 × Tversky loss (α=0.4, β=0.6)  
- **Optimizer:** Adam with OneCycleLR, AMP enabled  

---

## Training & evaluation

```bash
python train.py \
  --data_dir data/tiles \
  --epochs 450 --batch_size 8 \
  --lr 3e-4 --max_lr 1e-3 \
  --weight_decay 1e-5 \
  --pct_start 0.3 --div_factor 25 --final_div_factor 10000 \
  --loss focal_tversky --focal_alpha 0.6 --focal_gamma 2.0 \
  --tversky_alpha 0.4 --tversky_beta 0.6 \
  --amp --save_dir outputs/run_01
```

**Paper results:**  
- Accuracy: 89.55%  
- Precision: 0.6741  
- Recall: 0.6381  
- F1: 0.6556  
- IoU: 0.4876  
- Dice: 0.6556

---

## Inference & visualization

```bash
python infer.py \
  --scene_path data/raw/S1_scene.tif \
  --model_ckpt outputs/run_01/best.ckpt \
  --bands VV,VH,RATIO \
  --tile_size 382 --overlap 32 \
  --threshold 0.5 \
  --out_raster outputs/run_01/pred_illegal_logging.tif
```

Use `scripts/visualize.py` to overlay predictions on SAR inputs.

---

## Reproducibility & deployment

- **Seeds:** Set for Python, NumPy, Torch  
- **Logging:** Save config, commit hash, environment info  
- **Docker:** Build with CUDA runtime + dependencies for cloud portability  
- **Optional:** Deploy as a Gradio app on Hugging Face Spaces  

---

## Acknowledgments

This project is based on the research presented in:

- **Thinh Ha** – Beaver Works Summer Institute, Anaheim Discovery Christian School  
- **Tanish Khanna** – Beaver Works Summer Institute, La Jolla Country Day School  
- **Naga Kasam** – Beaver Works Summer Institute, Dr. Ronald E McNair Academic High School  
- **Arush Shangari** – Beaver Works Summer Institute, St. John's Preparatory School  
- **Ruhaan Arya** – Beaver Works Summer Institute, The Village School  
- **Ikshit Gupta** – Beaver Works Summer Institute, Mountain House High School  

Special thanks to Mr. Scheele (MIT Lincoln Lab), Mr. Amriche (SUNY), and Dr. Xiao (MIT Lincoln Lab) for their guidance, and to the Sentinel‑1/Copernicus program for providing SAR data.
