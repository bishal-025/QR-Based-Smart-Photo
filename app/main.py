from typing import List
from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .model import ImageClassifier
from .schemas import BatchPredictionResponse, ImagePrediction


MODEL_PATH = "models/best_model.pt"

app = FastAPI(
    title="Image Category Classifier",
    description="Classifies images into categories like lifestyle, academic, bar, etc.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


classifier: ImageClassifier | None = None


def get_classifier() -> ImageClassifier:
    global classifier
    if classifier is None:
        classifier = ImageClassifier(MODEL_PATH)
    return classifier


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=BatchPredictionResponse)
async def predict_images(files: List[UploadFile] = File(...)):
    clf = get_classifier()
    images: List[Image.Image] = []
    filenames: List[str] = []

    for file in files:
        contents = await file.read()
        img = Image.open(BytesIO(contents))
        images.append(img)
        filenames.append(file.filename)

    results = clf.predict_batch(images)

    predictions: List[ImagePrediction] = []
    for filename, result in zip(filenames, results):
        predictions.append(
            ImagePrediction(
                filename=filename,
                label=result["label"],
                confidence=result["confidence"],
                all_scores=result["all_scores"],
            )
        )

    return BatchPredictionResponse(predictions=predictions)


