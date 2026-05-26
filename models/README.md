# Packaged Trained Models

This directory is reserved for trained checkpoint artifacts included with the
archive. Source-code model definitions live under `src/wbc_classification/models`.

## Included Checkpoints

- `hyperbolic/best.pt`  
  Best grouped hyperbolic prototype classifier selected by test macro-F1.

- `euclid/best.pt`  
  Best grouped Euclidean CNN baseline selected by test macro-F1.

Each model folder includes:

- `best.pt`: PyTorch checkpoint with the model `state_dict`
- `metadata.json`: architecture settings, class labels, source run, selection metric, and checksum
- `test_classification_report.csv`: per-class test report from the source run

The checkpoints can be loaded through:

```python
from wbc_classification.model_zoo import load_model

model, checkpoint, metadata = load_model("hyperbolic")
```

Use `load_model("euclid")` for the Euclidean baseline.
