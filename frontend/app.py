import io
import os
from typing import List

import requests
from PIL import Image
import streamlit as st


# Allow overriding API URL via environment variable (for Docker, etc.)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")


def main():
    st.set_page_config(page_title="Image Category Classifier", layout="wide")
    st.title("Image Category Classifier")
    st.write(
        "Upload one or more images and the model will classify them into "
        "categories like **lifestyle**, **academic**, **bar**, etc."
    )

    uploaded_files = st.file_uploader(
        "Choose image files",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        return

    if st.button("Classify Images"):
        files = []
        for f in uploaded_files:
            # Convert to bytes-like object for requests
            bytes_data = f.read()
            files.append(("files", (f.name, bytes_data, "image/jpeg")))

        with st.spinner("Sending images to backend and getting predictions..."):
            try:
                response = requests.post(API_URL, files=files, timeout=60)
                response.raise_for_status()
            except Exception as e:
                st.error(f"Error calling API: {e}")
                return

        data = response.json()
        predictions: List[dict] = data.get("predictions", [])

        cols = st.columns(3)
        for idx, (file, pred) in enumerate(zip(uploaded_files, predictions)):
            col = cols[idx % len(cols)]
            with col:
                # Re-open image from bytes, because streamlit file_uploader resets pointer
                file.seek(0)
                img = Image.open(io.BytesIO(file.read()))
                st.image(img, caption=file.name, use_column_width=True)
                st.markdown(
                    f"**Predicted:** {pred['label']} "
                    f"(confidence: {pred['confidence']:.2f})"
                )

                with st.expander("Show scores for all categories"):
                    for score in pred.get("all_scores", []):
                        st.write(f"{score['label']}: {score['score']:.2f}")


if __name__ == "__main__":
    main()


