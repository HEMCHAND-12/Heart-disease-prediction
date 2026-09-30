"""Educational PaRSEL+ demo shell; no model is trained or saved by this file."""

import json
from pathlib import Path

MODEL_PATH = Path("models/parsel_plus_pipeline.joblib")
THRESHOLD_PATH = Path("models/parsel_plus_threshold.json")


def main() -> None:
    import joblib
    import streamlit as st

    st.set_page_config(page_title="PaRSEL+ educational demo")
    st.title("PaRSEL+ diabetes-risk demo")
    st.warning("For educational use only. This is not a medical diagnosis.")
    st.caption("TODO: train, validate, calibrate, and save the final frozen pipeline and threshold before enabling predictions.")
    if not MODEL_PATH.exists() or not THRESHOLD_PATH.exists():
        st.info("The validated model artifact is not available yet.")
        return
    _pipeline = joblib.load(MODEL_PATH)
    threshold = float(json.loads(THRESHOLD_PATH.read_text())["threshold"])
    st.write(f"Frozen decision threshold: {threshold:.4f}")
    st.write("Model loading is ready; input controls will be enabled after E6/E7 validation.")


if __name__ == "__main__":
    main()
