# Packaged Model Cards

## Hyperbolic Prototype Classifier

- **Path:** `models/hyperbolic/best.pt`
- **Architecture:** ResNet-18 backbone, linear projection, Poincare-ball hyperbolic prototype head.
- **Selection criterion:** Highest grouped-split test macro-F1 in the tuned hyperbolic sweep.
- **Test accuracy:** 0.8272
- **Test macro-F1:** 0.8244446620810439
- **Learned curvature magnitude:** 0.04356047511100769
- **Learned temperature:** 0.3162989318370819
- **Classes:** 14 harmonized leukocyte categories.
- **Intended use:** Research benchmarking and reproducibility of the associated manuscript.
- **Not intended for:** Clinical decision-making or patient diagnosis without regulatory validation.

## Euclidean CNN Baseline

- **Path:** `models/euclid/best.pt`
- **Architecture:** ResNet-18 backbone, linear projection, Euclidean linear classification head.
- **Selection criterion:** Highest grouped-split test macro-F1 in the Euclidean sweep.
- **Test accuracy:** 0.828954815864563
- **Test macro-F1:** 0.8190184078429702
- **Classes:** 14 harmonized leukocyte categories.
- **Intended use:** Research benchmarking and comparison with the hyperbolic classifier.
- **Not intended for:** Clinical decision-making or patient diagnosis without regulatory validation.

## Shared Caveats

The packaged checkpoints were trained on the harmonized ImageFolder benchmark described in the manuscript and deposited at Zenodo: https://doi.org/10.5281/zenodo.19126698. Image data are intentionally excluded from this Git repository. Performance should be interpreted under the acquisition-aware grouped split protocol documented in `docs/reproducibility.md`.
