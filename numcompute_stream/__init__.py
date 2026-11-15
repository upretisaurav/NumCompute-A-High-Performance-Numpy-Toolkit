"""
NumCompute-Stream: streaming machine learning with NumPy, built on the
2.1 NumCompute toolkit. Plots live in numcompute_stream.visualise (not
imported here, so matplotlib is only needed for plotting).
"""

from .io import iter_chunks, load_csv, read_csv_chunks
from .metrics import (
    F1,
    Accuracy,
    ConfusionMatrix,
    Precision,
    Recall,
    RollingMetric,
    StreamingAUC,
)
from .pipeline import Pipeline
from .preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler
from .stats import StreamingStats
from .stream import StreamTrainer
from .tree import DecisionTreeClassifier

__all__ = [
    "load_csv",
    "iter_chunks",
    "read_csv_chunks",
    "DecisionTreeClassifier",
    "StandardScaler",
    "MinMaxScaler",
    "OneHotEncoder",
    "SimpleImputer",
    "StreamingStats",
    "ConfusionMatrix",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "RollingMetric",
    "StreamingAUC",
    "Pipeline",
    "StreamTrainer",
]
