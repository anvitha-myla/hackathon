"""Data ingest, schema, splits, and causal cleaning (Prompt 1–3)."""

from src.data.cleaning import clean_batch, clean_series, load_cleaning_params
from src.data.quality import QualityFlag

__version__ = "0.0.0"
