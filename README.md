## Image Category Classifier (Lifestyle / Academic / Bar / etc.)

This project is a **full machine learning pipeline and web app** where you can:

- Upload **one or many images** from your computer
- Have a trained model automatically classify each image into categories such as **lifestyle**, **academic**, **bar**, and any other labels you define

### Tech Stack

- **Python 3.10+**
- **PyTorch** + **torchvision** (transfer learning with a pretrained CNN)
- **FastAPI** for the backend API
- **Uvicorn** as the development server
- **Streamlit** as a simple, modern web frontend

### Project Structure

```text
.
├── app
│   ├── main.py          # FastAPI app (REST API for inference)
│   ├── model.py         # Model definition, loading, and prediction helpers
│   └── schemas.py       # Pydantic models for API requests/responses
├── frontend
│   └── app.py           # Streamlit UI for image upload & visualization
├── training
│   ├── train.py         # Training script using transfer learning
│   └── config.yaml      # Config for training (paths, categories, hyperparams)
├── data
│   ├── raw              # Your raw images (organized in subfolders by category)
│   └── processed        # Train/val splits, if needed
├── models
│   └── best_model.pt    # Saved trained model weights (created after training)
├── requirements.txt
└── README.md
```

### Quick Start

1. **Create and activate a virtual environment (recommended)**

```bash
cd "Specialized Study/Specialized_Study_Project"
python -m venv .venv
source .venv/bin/activate  # on macOS/Linux
# .venv\Scripts\activate   # on Windows PowerShell
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Prepare your dataset**

Place your labeled images under `data/raw` in subfolders, one folder per category.  
For example:

```text
data/raw/
├── lifestyle/
│   ├── img1.jpg
│   ├── img2.png
│   └── ...
├── academic/
│   ├── img3.jpg
│   └── ...
├── bar/
│   ├── img4.jpg
│   └── ...
└── other_category/
    └── ...
```

4. **Configure labels and hyperparameters**

Edit `training/config.yaml` to:

- Add/remove **categories**
- Adjust **batch size**, **learning rate**, **number of epochs**
- Set **train/validation split**

5. **Train the model**

```bash
python -m training.train
```

This will:

- Use a pretrained CNN (e.g. ResNet18) from `torchvision`
- Fine-tune it on your dataset
- Save the best performing weights to `models/best_model.pt`

6. **Run the backend API**

```bash
uvicorn app.main:app --reload
```

The API will start at `http://127.0.0.1:8000`.

7. **Run the frontend**

In another terminal (with the same virtual environment activated):

```bash
streamlit run frontend/app.py
```

This opens a web app in your browser where you can upload one or many images and see predictions.

### Next Steps / Customization

- Add more categories by:
  - Creating new folders under `data/raw`
  - Updating the `class_names` / categories in `training/config.yaml`
- Improve performance by:
  - Using a deeper model (e.g. ResNet50, EfficientNet)
  - Collecting more training data
  - Tuning hyperparameters
- Deploy to production:
  - Containerize with Docker
  - Host the FastAPI app on a cloud provider
  - Host the Streamlit app or build a custom React frontend

You can ask the assistant to customize any part of this stack (e.g. different framework, different model, or integration with an existing system).

