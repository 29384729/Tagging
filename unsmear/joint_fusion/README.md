# Joint Fusion Unsmear

This folder contains the joint-fusion unsmearing experiment. The main entry point is `unsmear.ipynb`. The `tool/` folder holds the supporting model, data, training, metric, analysis, and I/O helper modules used by the notebook.

## Model Design

The joint model is built around one shared particle-token encoder. The HLT-like jet features are first embedded and passed through the shared Transformer encoder, producing a common token representation for both reconstruction and classification.

After the shared encoder, the model has two task heads:

- An unsmear head reconstructs offline-like particle features from the HLT input. It uses a small Transformer decoder followed by a regression head and predicts a per-particle correction `delta`; the reconstructed view is `x + delta`.
- A classifier head pools the shared encoder tokens into a jet-level representation and predicts the signal/background logit.

The fusion version connects these two heads. The reconstruction correction is projected into the classifier embedding space, pooled into `z_delta`, and combined with the normal classifier embedding `z_main`:

```text
z_final = z_main + alpha * gate(z_main, z_delta) * z_delta
```

Here `gate` is a learned feature-wise gate and `alpha` is a learned positive scale. In this folder, `cls_use_delta_fusion=True`, so the classifier uses the learned unsmearing correction. Also, `cls_detach_delta_for_cls=False`, so the classification loss backpropagates through the fusion path into the reconstruction branch.

## Training Design

The notebook trains and compares offline teacher, HLT student, HLT+KD student, and joint models over repeated seeds. The joint models optimize a mixed objective:

- Reconstruction loss between the joint reconstructed particles and the offline target.
- Hard classification loss on labels.
- Optional KD loss from the offline teacher.
- Optional teacher-embedding and physics-consistency terms, controlled by notebook config weights.

The training loader can resmear the input jets every epoch with epoch-dependent random seeds. When `train_resmear_repeats > 1`, each original jet is repeated within the epoch and each repeat gets its own HLT-like smeared view. All repeats keep the same offline target, label, and sample weight.

The baseline HLT/KD models and the joint models use the same shared backbone configuration where possible, so performance differences are mainly driven by the joint unsmearing objective and the delta-fusion path.

## Technical Analysis

The notebook includes fusion-specific diagnostics:

- Gate and alpha curves over training epochs.
- Fusion-ratio rows and summary tables for `||alpha * gate * z_delta|| / ||z_main||`.
- Reconstruction residual comparison for the best joint model.
- Standard AUC, FPR@TPR, and gap-recovery summaries for HLT, KD, and joint-fusion variants.

## Outputs

`unsmear.ipynb` writes run artifacts under `runs/<RUN_NAME>/`, including checkpoints, repeat metrics, prediction bundles, figures, and tables. The summary tables include AUC, weighted AUC, FPR at target TPR values, gap-recovery metrics, and fusion diagnostics such as gate/alpha curves and the `||alpha * gate * z_delta|| / ||z_main||` ratio.