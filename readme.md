# ATLAS Top Tagging Smearing Study

This repository studies the [ATLAS top tagging open data](https://gitlab.cern.ch/atlas/ATLAS-top-tagging-open-data) task under detector and trigger resolution effects. The baseline problem is to classify boosted top-quark jets against background jets using constituent-level jet information.

The study builds HLT-like inputs by applying Gaussian smearing to offline-quality constituents. 

The main goal is to improve top-quark and background-jet classification on HLT-like data through resolution recovery of the smeared HLT-like inputs. The offline-trained and HLT-trained classifiers are used as the best and worst baselines, and the notebooks record how the other models try to use resolution recovery to obtain better classification results on HLT-like data.

Deprecated directories are kept as earlier experiments and implementation references. The active analysis is organized under `unsmear`, with separate variants for separated training, joint training without delta fusion, and joint training with delta fusion.
