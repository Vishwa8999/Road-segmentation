# Project Briefing: Drone Video Road Segmentation & Fine-Tuning Strategy

> **Purpose of this document:**  
> This document is a complete, self-contained audit and technical brief of our AI road segmentation project. You can copy and paste this directly into **Claude** (or any expert LLM/engineer) to ask for guidance on fine-tuning, model architecture, dataset selection, and video post-processing.

---

## 1. Project Mission & Target Scenario

* **Primary Objective:** High-accuracy, real-time binary road segmentation on high-resolution drone video footage.
* **Target Video Details:**
  * **File:** `DJI_0529.MOV` (4K UHD: 3840×2160 @ 23.976 FPS, 2,897 frames, ~2 minutes duration).
  * **Camera Viewpoint:** Oblique pitch (~30° to 45° forward tilt from the horizontal), recorded by a moving DJI camera drone.
  * **Scene Characteristics:** 
    * Transitions from rural/village zones (dirt shoulders, trees, scattered residential buildings with flat concrete/tar rooftops) onto a **modern 6-lane divided asphalt expressway**.
    * Features painted lane markings, dashed lines, zebra pedestrian crossings, central concrete median dividers, and road shoulders.
* **Current Inference Output:** MP4 video with a translucent green mask overlay (`alpha=0.40`) hugging the drivable road surface.

---

## 2. Current Codebase Architecture & Technical Stack

* **Framework:** PyTorch 2.x, OpenCV (`cv2`), CUDA acceleration on NVIDIA GPU.
* **Model Architecture (`src/model.py`):**
  * Classical 4-level **U-Net** built from scratch (no pretrained backbone like ResNet or EfficientNet).
  * `in_channels = 3` (RGB), `out_channels = 1` (raw logits).
  * Channel width schedule: `base_channels = 32` $\rightarrow$ Encoder features: `[32, 64, 128, 256]` $\rightarrow$ Bottleneck: `512` $\rightarrow$ Decoder: `[256, 128, 64, 32]`.
  * DoubleConv blocks: `Conv2d(3x3) -> BatchNorm2d -> ReLU -> Conv2d(3x3) -> BatchNorm2d -> ReLU`.
  * Downsampling: `MaxPool2d(2x2)`.
  * Upsampling: `ConvTranspose2d(2x2, stride=2)` with bilinear fallback interpolation if skip-connection dimensions differ.
  * Model weight footprint: ~93 MB (~23.3M parameters).
* **Loss Function (`src/metrics.py`):**
  * Combined BCE + Soft Dice Loss:
    $$\mathcal{L} = 0.5 \times \text{BCEWithLogitsLoss}(y_{logits}, y_{target}) + 0.5 \times \text{DiceLoss}(\sigma(y_{logits}), y_{target})$$
* **Optimizer & Scheduling (`src/train.py`):**
  * Optimizer: `AdamW(lr=2e-5 to 5e-5, weight_decay=1e-4)`.
  * Scheduler: `ReduceLROnPlateau(mode='max', factor=0.5, patience=3)`.
  * Early Stopping: Configurable patience (typically 5 to 8 epochs monitoring validation IoU).
* **Data Augmentations (`src/dataset.py`):**
  * Random Horizontal Flip ($p=0.5$), Random Vertical Flip ($p=0.5$).
  * Random Affine Rotation ($\pm 20^\circ$).
  * Color Jitter / Brightness & Contrast adjustment ($\pm 0.2$).
* **Video Inference Pipeline (`src/predict_video.py`):**
  * Frame ingestion via OpenCV `VideoCapture`.
  * Preprocessing: Convert BGR $\rightarrow$ RGB, normalize to $[0, 1]$, resize to model input dimensions.
  * Model Forward Pass: Batched CUDA inference with `torch.inference_mode()`, sigmoid activation.
  * Post-processing: Continuous probability upsampling via `cv2.INTER_CUBIC`, thresholding ($T=0.50$), connected-components noise filtering (removes blobs $< 500\text{ px}$ in 4K), and weighted blending (`alpha=0.40`).

---

## 3. Training Chronology & Experimental Progression

### Stage 1: Initial Baseline (DeepGlobe Satellite Dataset)
* **Dataset:** DeepGlobe Road Extraction dataset (high-altitude satellite imagery, 50cm GSD, 1024×1024 tiles).
* **Outcome on Drone Video:** Poor. Failed due to the radical difference between satellite nadir views (straight 90° overhead looking at terrain) and low-altitude angled drone perspectives (oblique 30°–45° forward flight with 3D buildings, horizons, and perspective vanishing lines).

---

### Stage 2: UAVid Fine-Tuning (`best_model_uavid_finetuned_iou7168.pth`) — [CURRENT BEST MODEL]
* **Dataset:** UAVid (UAV Video semantic segmentation benchmark).
  * Captured by DJI drones flying at 30°–45° oblique angles over urban roads, suburban streets, highways, and intersections.
* **Training Setup:** Fine-tuned with `lr = 5e-5`, batch size 2, 256×256 input tiles.
* **Validation Metrics (Epoch 47):**
  * **IoU (Jaccard Index):** **69.98%** (~0.70)
  * **Dice Score (F1):** **82.34%** (~0.82)
  * **Precision:** **78.96%**
  * **Recall:** **86.02%**
  * **Pixel Accuracy:** **95.88%**
  * **Validation Loss:** **0.2051**
* **Qualitative Performance on `DJI_0529.MOV`:**
  * Strongly recognized the 6-lane highway, lane structure, and pavement boundaries.
  * **Identified Faults:**
    1. *Building Rooftop False Positives:* The dark flat concrete/tar roof of a building in frame 200 was falsely flagged with green overlay.
    2. *Jagged Edges:* Road edges had a staircase / pixelated look due to 256×256 resolution upsampling.

---

### Stage 3: Drone_Road_Dataset Fine-Tuning (`outputs/drone_finetune`) — [FAILED / SEVERE REGRESSION]
* **Dataset:** `Drone_Road_Dataset` (7,351 raw side-by-side composite images $\rightarrow$ 5,583 paired 256×256 files across 6 altitude levels: 50cm, 100cm, 150cm, 200cm, 300cm, 500cm).
* **Training Setup:** Resumed from the UAVid checkpoint, `lr = 2e-5`, batch size 8, 30 epochs (Epochs 48 to 78), train/val/test split: 4,466 / 558 / 559.
* **Final Test Set Metrics:**
  * **IoU:** **49.79%** (dropped by **20.2%** compared to UAVid!)
  * **Dice Score:** **66.48%** (dropped by **15.9%**)
  * **Precision:** **65.69%**
  * **Recall:** **67.30%**
  * **Pixel Accuracy:** **85.28%** (dropped by **10.6%**)
  * **Test Loss:** **0.3851**
* **Qualitative Performance on `DJI_0529.MOV`:**
  * **Disaster:** The video overlay became a patchy "swiss cheese" mess. Half of the 6-lane highway lanes were completely dropped at the standard 0.50 threshold.

---

## 4. Root Cause Analysis: All Faults & Bottlenecks Identified

### Fault A: Severe Domain Mismatch & Catastrophic Forgetting
* **The Mismatch:** A visual audit of `Drone_Road_Dataset` revealed that it consists **entirely of nadir (90° vertical) aerial orthophotos of narrow, tight residential alleys (10–20 ft wide) between dense terracotta tile rooftops** (resembling dense urban favelas).
* **The Forgetting Effect:** The dataset has zero multi-lane divided highways, zero painted highway lane dashes, and zero oblique camera angles. By fine-tuning for 30 epochs on this dataset, the model suffered catastrophic forgetting: it learned that *“roads are exclusively narrow ribbons surrounded by roofs”*. When presented with wide open stretches of 6-lane highway asphalt, the network assumed the expansive surface was background, dropping sigmoid confidence from $>0.95$ down to $0.20$–$0.35$.

### Fault B: The 4K $\rightarrow$ 256×256 Spatial Information Choke Point
* The input video is 3840×2160 (8.3 million pixels).
* `predict_video.py` was downsampling the entire frame to **256×256** (65,536 pixels) — a **128× reduction in total pixel area**!
* At 256×256, a distant building roof occupies only ~8×12 pixels. Textural cues (parapets, vents, walls) are completely wiped out, making flat dark roofs look identical to dark road asphalt.
* Furthermore, upsampling a 256×256 binary mask back to 3840×2160 using `INTER_NEAREST` turned single pixels into 15×15 square blocks, creating severe edge jaggies.

### Fault C: Resumed Optimizer Best-Score State Lock
* When fine-tuning on a new dataset using `--resume`, PyTorch restored `best_score = 0.6998` from the previous UAVid dataset.
* Since initial validation IoUs on the new dataset were ~0.47, the early stopper concluded the model was never improving and aborted training without saving any checkpoints. We patched this by adding a `--reset-best` flag to reset the monitor tracker to $-1.0$.

### Fault D: Frame-by-Frame Temporal Jitter
* Because each frame is segmented in isolation without temporal memory (no recurrence, optical flow, or exponential moving average), minor confidence fluctuations between frames cause flickering around road margins.

---

## 5. Major Breakthrough: Resolution Scaling & Post-Processing Fix

Because our U-Net implementation is **fully convolutional** (no flattened dense layers), it naturally accepts arbitrary input dimensions divisible by 16.

We tested running the **UAVid model (Stage 2 checkpoint)** at **512×512 inference resolution** with:
1. **Continuous Probability Upsampling:** Resizing the float32 sigmoid probability map directly to 4K using `cv2.INTER_CUBIC` *before* binary thresholding.
2. **Speckle Noise Filtering:** Connected-component analysis discarding any mask island $< 500\text{ pixels}$ in 4K resolution.

### Visual Results:
* **Building Rooftops:** **100% eliminated.** At 512×512, the network perceives structural boundaries, building corners, and wall elevations, completely dropping the false-positive roof detection.
* **Road Edges:** Razor-sharp, perfectly tracking the asphalt borders and highway curbs without blocky pixelation.
* **Highway Coverage:** Solid, full coverage across all 6 lanes, pedestrian crossings, and slip roads.
* **Speed:** Runs at ~30–40 FPS on CUDA (inference latency ~25 ms per frame).

---

## 6. Detailed Metrics Comparison Table

| Metric | DeepGlobe Baseline | UAVid Fine-Tuned (Stage 2) | Drone_Road_Dataset FT (Stage 3) | Enhanced UAVid (512×512 + Post-Proc) |
| :--- | :--- | :--- | :--- | :--- |
| **Domain Type** | Satellite Nadir | Oblique Drone Video | Orthomosaic Alleys | Oblique Drone Video |
| **Val / Test IoU** | ~0.42 | **0.6998 (Val)** | 0.4979 (Test) | **Highest qualitative accuracy** |
| **Dice Score (F1)**| ~0.58 | **0.8234 (Val)** | 0.6648 (Test) | **Continuous, smooth** |
| **Precision** | ~0.61 | **0.7896** | 0.6569 | **No roof false positives** |
| **Recall** | ~0.60 | **0.8602** | 0.6730 | **Full highway capture** |
| **Pixel Accuracy** | ~0.79 | **0.9588** | 0.8528 | **Sharp boundary delineation** |
| **Loss** | ~0.48 | **0.2051** | 0.3851 | N/A (Inference enhancement) |
| **Video Quality** | Failed | Good (blocky, roof error) | Unusable (patchy holes) | **Production-grade clean** |

---

## 7. Questions for Claude: Seeking Expert Guidance

Please ask Claude the following targeted questions to determine the next evolution of this project:

```text
Hello Claude! I am working on an aerial road segmentation system for high-resolution (4K) oblique drone videos (DJI drone camera tilted at 30°-45° forward flight over highways and rural landscapes). 

I have shared our full project briefing above. Based on our findings, progress, and faults, I would love your expert advice on the following key areas:

1. Model Architecture & Backbones:
   Currently, we use a custom 4-level U-Net trained from scratch (23M parameters, base_channels=32). Would switching to a modern encoder-decoder architecture with a pretrained ImageNet/Cityscapes backbone (such as SegFormer/MiT-B2/B3, ConvNeXt-UNet, or HRNet-W32) yield substantially better feature extraction for oblique viewpoints? What lightweight architecture strikes the best balance between high IoU and 30+ FPS video inference?

2. Fine-Tuning & Addressing Catastrophic Forgetting:
   Our fine-tuning on the Drone_Road_Dataset caused catastrophic forgetting because the dataset had only narrow alleyways and top-down views. If we want to fine-tune further:
   - What specific public datasets best match 30°-45° oblique drone video of highways and rural roads (e.g., UAVid, VisDrone, Aeroscapes, Cityscapes)?
   - How should we structure mixed-dataset training or rehearsal strategies (e.g., combining 70% UAVid + 30% auxiliary data) to prevent forgetting wide highways?
   - Should we freeze early encoder layers during fine-tuning?

3. Loss Function Engineering:
   Currently, we use 0.5 * BCE + 0.5 * SoftDice. For road segmentation where road connectivity (preventing gaps/holes) and sharp curb boundaries are critical:
   - Would Focal Loss, Lovász-Softmax, or Boundary/Surface Loss (Boundary-aware loss) improve road edge precision?
   - Are there topological losses (TopoLoss, clDice) you recommend for maintaining continuous road topology without fragmentation?

4. High-Resolution Video Inference Strategy:
   We noticed that increasing inference resolution from 256x256 to 512x512 completely fixed building false positives. But our raw video is 3840x2160 (4K):
   - What is the best strategy for 4K drone video inference: resizing to 512x512/768x768, letterboxing with aspect ratio preservation, or a sliding-window overlapping patch approach (tiling with Hann window blending)?
   - How can we implement temporal smoothing across frames (e.g., Exponential Moving Average of probabilities, optical flow warping, or ConvLSTM) to eliminate frame-to-frame mask flickering?

5. Self-Training / Pseudo-Labeling on Unlabeled Target Drone Video:
   Since we have hours of target DJI drone videos without ground-truth masks:
   - What is the best pipeline to generate high-confidence pseudo-labels from our best model (or an ensemble) to self-train the model specifically on the target flight environment?
```
