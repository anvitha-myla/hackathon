"""A vs B vs C evaluation (Prompt 10) and stress tests (Prompt 11).

Do not assume the hybrid wins.
"""

__version__ = "0.0.0"

from src.evaluation.metrics import (
    MODEL_HYBRID,
    MODEL_MECHANISTIC,
    MODEL_NN_ONLY,
    MODEL_ORDER,
    mae,
    r2_score,
    rmse,
)
from src.evaluation.runner import (
    EvaluationRun,
    TestSplitLeakageError,
    evaluate_batches,
    run_final_test,
)

__all__ = [
    "MODEL_HYBRID",
    "MODEL_MECHANISTIC",
    "MODEL_NN_ONLY",
    "MODEL_ORDER",
    "EvaluationRun",
    "TestSplitLeakageError",
    "evaluate_batches",
    "mae",
    "r2_score",
    "rmse",
    "run_final_test",
]
