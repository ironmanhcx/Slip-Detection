# Contact Status Recognition and Slip Detection with a Bio-inspired Tactile Hand

This repository contains the code for our tactile slip detection framework and the pneumatic control program for the bio-inspired dexterous hand proposed in the paper **“Contact Status Recognition and Slip Detection with a Bio-inspired Tactile Hand”**. The project includes:

- **MATLAB code** for tactile signal processing, contact status recognition, and slip onset detection
- **LabVIEW code** for pneumatic actuation/control of the dexterous hand

The proposed method is built on a **five-fingered bio-inspired tactile hand** with **24 tactile channels** in total, including **14 strain-gauge (SG) channels** for static force perception and **10 PVDF channels** for dynamic/slip-related vibration perception. The overall pipeline performs **signal preprocessing**, **binning**, **feature extraction**, **feature selection**, and **kernel-ELM-based contact status classification**, then estimates the **slip onset time** from the predicted contact state sequence. In the paper, the method achieved **96.39%** test accuracy on seen materials and **91.95%** accuracy on unseen materials.

## Repository Structure

```
Github_code/
├── Framework of the Tactile Signal Processing Approach/
│   ├── confusion matrix of seen materials.m
│   ├── datapatition_label_eaugmentation.m
│   ├── Drawing of feature statistics.m
│   ├── elm_kernel_test.m
│   ├── elm_kernel_train.m
│   ├── extract_features_1signal.m
│   ├── model_test_unseenmaterial.m
│   ├── sliptime_prediction10M.m
│   ├── stage1_build_featurepool.m
│   └── valid_data_extraction.m
│
└── LabVIEW Pneumatic Control/
    └── pneumatic control.vi
```

## Project Overview

Robotic grasping of fragile or smooth objects requires accurate detection of incipient slip so that the grip force can be adjusted in time. Instead of using a simple threshold-based slip detector, this work reformulates slip detection as a **contact status recognition** problem:

- **Non-slip**
- **Slip**

After classifying each short time segment of tactile data, the slip onset time can be determined from the transition between the two states.

## Method Pipeline

The tactile signal processing framework follows the pipeline below:

1. **Signal acquisition** from 24 tactile channels
2. **Signal preprocessing** using Savitzky–Golay filtering
3. **Binning** of time-series tactile signals into short segments
4. **Feature extraction** in time and frequency domains
5. **Feature selection** from a large feature pool
6. **Kernel Extreme Learning Machine (kernel-ELM)** training/testing
7. **Slip onset detection** based on the predicted contact status sequence

According to the paper, the framework uses:

- **24 tactile channels** from the five-fingered tactile hand
- **Savitzky–Golay filtering** for denoising
- **50 ms bin width**
- **Discrete Wavelet Transform (DWT)** and **FFT**
- **17 handcrafted features**
- **120 selected optimal features**
- **Polynomial kernel ELM** for classification

## Experimental Setup

The tactile hand is mounted on a robotic arm and interacts with cylindrical objects under different sliding conditions. The dataset in the paper was collected using:

- **6 training materials**: ABS, aluminum foil, fiberglass aluminum foil, oil paper, acrylic, PVC
- **4 unseen materials** for generalization test: wood, iron, copper, cloth
- **3 sliding velocities**: 20 mm/s, 40 mm/s, 60 mm/s
- **28 trials per material condition**

This produces a dataset of **504 trials** for the main experiments.

## Main Files

Below is a practical description of the scripts based on their filenames and the paper workflow:

### MATLAB: Tactile Signal Processing Framework

- `valid_data_extraction.m`
   Extracts valid tactile data segments from raw experiments.
- `datapatition_label_eaugmentation.m`
   Performs data partitioning, labeling, and possible augmentation.
- `stage1_build_featurepool.m`
   Builds the full feature pool from tactile signal bins.
- `extract_features_1signal.m`
   Extracts features from one tactile signal segment/bin.
- `elm_kernel_train.m`
   Trains the kernel-ELM contact status recognition model.
- `elm_kernel_test.m`
   Tests the trained model on seen-material data.
- `model_test_unseenmaterial.m`
   Evaluates generalization performance on unseen materials.
- `confusion matrix of seen materials.m`
   Plots the confusion matrix for the seen-material test set.
- `Drawing of feature statistics.m`
   Draws/visualizes the statistics of selected features.
- `sliptime_prediction10M.m`
   Predicts slip onset time from recognition results.

### LabVIEW: Pneumatic Control

- `pneumatic control.vi`
   LabVIEW program for pneumatic actuation/control of the dexterous tactile hand.

## Recommended Workflow

A suggested running order is:

```
1. valid_data_extraction.m
2. datapatition_label_eaugmentation.m
3. stage1_build_featurepool.m
4. extract_features_1signal.m
5. elm_kernel_train.m
6. elm_kernel_test.m
7. model_test_unseenmaterial.m
8. confusion matrix of seen materials.m
9. Drawing of feature statistics.m
10. sliptime_prediction10M.m
```

> Note: Please modify file paths, data directories, and parameter settings according to your local environment before running the scripts.

## Requirements

### MATLAB

Recommended:

- MATLAB R2021a or later
- Signal processing / wavelet related functions available
- Standard plotting functions for visualization

### LabVIEW

- LabVIEW environment compatible with `.vi` files
- Proper DAQ / pneumatic control hardware configuration if running on the physical system

## Results

Reported performance in the paper:

- **Contact status recognition accuracy on seen materials:** **96.39%**
- **Recognition accuracy on unseen materials:** **91.95%**

The method shows good generalization across different materials and sliding velocities, demonstrating its potential for practical slip detection in robotic grasping.

## Notes

- The MATLAB scripts in this repository correspond to the tactile signal processing framework described in the paper.
- The LabVIEW file is used for the pneumatic control side of the hardware system.
- Some scripts may require custom data files that are not included in this repository. Please organize your raw/processed data paths accordingly.
- File names are kept close to the original experimental code for consistency with the research workflow.

## Citation

If you find this repository useful, please cite the corresponding paper :

```
@article{he2026contact,
  title={Contact Status Recognition and Slip Detection with a Bio-inspired Tactile Hand},
  author={He, Chengxiao and Yang, Wenhui and Zhao, Hongliang and Lv, Jiacheng and Shao, Yuzhe and Qin, Longhui},
  journal={arXiv preprint arXiv:2603.18370},
  year={2026}
}
```

