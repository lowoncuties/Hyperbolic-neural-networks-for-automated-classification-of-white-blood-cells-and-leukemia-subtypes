"""Model builders used by the WBC classification trainers."""

from .classic_cnn import CNNClassifier
from .hyperbolic_cnn import HyperbolicClassifier, HyperbolicPrototypeHead

__all__ = ["CNNClassifier", "HyperbolicClassifier", "HyperbolicPrototypeHead"]
