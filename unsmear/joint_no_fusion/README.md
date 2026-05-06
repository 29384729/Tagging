# Joint No-Fusion Unsmear

This folder contains the no-fusion joint unsmearing experiment. The main entry point is `unsmear.ipynb`. The `tool/` folder holds the supporting model, data, training, metric, analysis, and I/O helper modules used by the notebook.

## Model Design

The joint model is built around one shared particle-token encoder. The HLT-like jet features are first embedded and passed through the shared Transformer encoder, producing a common token representation for both reconstruction and classification.

After the shared encoder, the model has two task heads:

- An unsmear head reconstructs offline-like particle features from the HLT input. It uses a small Transformer decoder followed by a regression head and predicts a per-particle correction `delta`; the reconstructed view is `x + delta`.
- A classifier head pools the shared encoder tokens into a jet-level representation and predicts the signal/background logit.

The no-fusion version keeps these two heads connected only through the shared encoder. The classifier uses the pooled encoder representation `z_main` directly:

```text
z_final = z_main
```

In this folder, `cls_use_delta_fusion=False`, so the reconstructed correction `delta` is not projected back into the classifier embedding. Also, `cls_detach_delta_for_cls=True`, so the classifier loss does not use the reconstruction branch as a fusion path. The no-fusion notebook focuses on how the reconstruction head changes the shared encoder through its gradients.

## Training Design

The notebook trains and compares offline teacher, HLT student, HLT+KD student, and joint models over repeated seeds. The joint models optimize a mixed objective:

- Reconstruction loss between the joint reconstructed particles and the offline target.
- Hard classification loss on labels.
- Optional KD loss from the offline teacher.
- Optional physics-consistency terms, controlled by notebook config weights.

The training loader can resmear the input jets every epoch with epoch-dependent random seeds. Each resmeared HLT-like view keeps the same offline target, label, and sample weight.

The baseline HLT/KD models and the joint models use the same shared backbone configuration where possible, so performance differences are mainly driven by the shared-encoder joint objective without the delta-fusion path.

## Technical Analysis

The notebook includes extra analysis sections for checking how the reconstruction objective affects the shared encoder:

- Reconstruction residual comparison for the best joint model.
- Gradient norm and gradient-cosine probes for reconstruction, hard classification, and KD losses on shared encoder parameter groups.
- Feature-level reconstruction gradient probes for the shared encoder.
- Teacher embedding distance comparisons for HLT, joint no-KD, and joint with-KD reconstructions.
- Case studies where HLT or joint predictions disagree with the offline teacher.
- Correlation checks between teacher embedding distance and logit gap.

## Outputs

`unsmear.ipynb` writes run artifacts under `runs/<RUN_NAME>/`, including checkpoints, repeat metrics, prediction bundles, gradient probe tables, figures, and summary tables. The summary tables include AUC, weighted AUC, FPR at target TPR values, gap-recovery metrics, and shared-encoder gradient summaries for comparing HLT, KD, and no-fusion joint variants.
