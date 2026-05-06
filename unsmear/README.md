# Unsmear Experiments

This folder contains three unsmearing experiment tracks. Each subfolder has its own README with notebook-level details.

## Folders

- `seperated/`: trains upstream unsmear models first, then trains downstream taggers on HLT, reconstructed, and KD feature views.
- `joint_no_fusion/`: trains a shared-encoder joint reconstruction/classification model without feeding the reconstruction correction back into the classifier. This track focuses on shared-encoder gradient analysis.
- `joint_fusion/`: trains the joint model with delta fusion, where the learned reconstruction correction contributes to the classifier representation through gate and alpha terms.

## Main Entry Points

Each track uses `unsmear.ipynb` as the main running notebook. The `tool/` folder inside each track contains the helper modules used by that track.
