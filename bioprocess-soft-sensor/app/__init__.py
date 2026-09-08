"""Streamlit four-tab application package (docs/12_UI_SPEC.md)."""

__version__ = "0.0.0"


def main() -> None:
    from app.streamlit_app import main as run_app

    run_app()
