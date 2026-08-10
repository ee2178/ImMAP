"""Task- and metric-agnostic evaluation sweep.

    evaluation.metrics   what to measure  (registry, per-sample, config-toggled)
    evaluation.tasks     how to run one batch for a given cfg["task"]
    scripts/evaluate.py  the CLI that walks trained runs and writes a CSV
"""

from evaluation.metrics import DEFAULT_METRICS, REGISTRY as METRIC_REGISTRY
from evaluation.metrics import build_metrics, warm_up
from evaluation.tasks import REGISTRY as TASK_REGISTRY
from evaluation.tasks import build_adapter, default_sigma

__all__ = ["METRIC_REGISTRY", "TASK_REGISTRY", "DEFAULT_METRICS",
           "build_metrics", "build_adapter", "default_sigma", "warm_up"]
