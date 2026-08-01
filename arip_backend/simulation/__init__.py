"""ARIP digital-twin simulation package."""

from arip_backend.simulation.ode_engine import ODEEngine, ODEEngineConfig

__all__ = ["ODEEngine", "ODEEngineConfig", "run_pipeline"]


def run_pipeline(*args, **kwargs):
    """Lazy wrapper to avoid circular imports when used as ``python -m``."""
    from arip_backend.simulation.pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)
