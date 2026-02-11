from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class ImageClassifier:
    def __init__(self, model_path: str, device: str | None = None):
        self.model_path = Path(model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Train the model first with `python -m training.train`."
            )

        checkpoint = torch.load(self.model_path, map_location=self.device)
        class_names: List[str] = checkpoint["class_names"]
        self.class_names = class_names

        num_classes = len(class_names)
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        self.model = model

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def predict_image(self, image: Image.Image) -> dict:
        img = image.convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            prob_values = probs.tolist()

        max_idx = int(torch.argmax(probs).item())
        predicted_label = self.class_names[max_idx]
        confidence = prob_values[max_idx]

        return {
            "label": predicted_label,
            "confidence": float(confidence),
            "all_scores": [
                {"label": label, "score": float(score)}
                for label, score in zip(self.class_names, prob_values)
            ],
        }

    def predict_batch(self, images: List[Image.Image]) -> List[dict]:
        results: List[dict] = []
        for img in images:
            results.append(self.predict_image(img))
        return results


