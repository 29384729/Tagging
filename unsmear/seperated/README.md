# Separated Unsmear

This folder runs the separated unsmearing workflow. The main notebook is `unsmear.ipynb`. The `tool/` folder contains the model, data, training, metric, flow-matching, preprocessing, and I/O helper modules used by the notebooks.

## Model Design

The workflow trains an upstream reconstruction model first, then trains downstream taggers on reconstructed features.

The upstream unsmear model takes HLT-like smeared particle features and reconstructs offline-like particle features. The integrated notebook runs several upstream reconstruction models:

- MLP token regressor.
- Transformer token regressor.
- Flow Matching model.

The downstream tagger is trained after reconstruction. It uses feature views such as the original HLT view, the reconstructed view from an upstream unsmear model, and KD variants. The offline teacher, HLT, and HLT+KD baselines provide the downstream references.

## Training Design

The notebook builds offline and HLT feature views from the raw HDF5 constituents, trains upstream reconstruction models, and evaluates downstream taggers over repeat seeds.

The upstream training loader can resmear the input jets every epoch with epoch-dependent random seeds. Each resmeared HLT-like view keeps the same offline reconstruction target and mask.

For downstream evaluation, `downstream_repeat_seeds` controls the repeated downstream tagger runs. `kd_teacher_repeat_seed` selects the repeat used as the downstream KD teacher. When external baselines are disabled, the notebook trains local offline, HLT, and HLT+KD baselines and uses those local checkpoints for downstream comparisons.

## Analysis

The integrated notebook reports:

- Upstream reconstruction quality for MLP, Transformer, and Flow Matching models.
- Downstream AUC and weighted AUC for HLT, reconstructed, and KD variants.
- FPR at target TPR values.
- Gap recovery relative to the offline teacher and HLT baseline.
- Residual distributions and reconstruction summary tables.

Single-model notebooks such as `unsmear_mlp.ipynb`, `unsmear_transformer.ipynb`, `unsmear_fm.ipynb`, and `unsmear_U-Net.ipynb` keep narrower versions of the same separated workflow. `smeareffect_exploration.ipynb` is used for smear-effect scans and single-feature perturbation studies.

## Outputs

`unsmear.ipynb` writes run artifacts under `runs/<RUN_NAME>/`, including upstream checkpoints, downstream checkpoints, repeat metrics, prediction bundles, figures, and tables. The summary tables include AUC, weighted AUC, FPR@TPR, and gap-recovery metrics for HLT, offline references, and separated unsmear variants.
