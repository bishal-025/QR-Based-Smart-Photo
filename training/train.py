import os
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms

import yaml


def load_config(config_path: str = "training/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    return train_transform, val_transform


def create_dataloaders(cfg: dict):
    data_dir = cfg["data_dir"]
    batch_size = cfg["training"]["batch_size"]
    val_split = cfg["training"]["val_split"]
    num_workers = cfg["training"]["num_workers"]
    image_size = cfg["training"]["image_size"]

    train_transform, val_transform = get_transforms(image_size)

    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)

    if len(full_dataset.classes) == 0:
        raise RuntimeError(
            f"No classes found in {data_dir}. "
            "Make sure you have subfolders like 'lifestyle', 'academic', 'bar', etc."
        )

    n_total = len(full_dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
    # Override transform for validation subset
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, full_dataset.classes


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(model, criterion, optimizer, dataloader, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        total += inputs.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = running_corrects / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def evaluate(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total += inputs.size(0)

    epoch_loss = running_loss / total if total > 0 else 0.0
    epoch_acc = running_corrects / total if total > 0 else 0.0
    return epoch_loss, epoch_acc


def main():
    cfg = load_config()
    set_seed(cfg.get("random_seed", 42))

    output_model_path = cfg["output_model_path"]
    Path(os.path.dirname(output_model_path)).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = create_dataloaders(cfg)
    print(f"Found classes: {class_names}")

    num_classes = len(class_names)
    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    num_epochs = cfg["training"]["num_epochs"]

    best_val_acc = 0.0
    best_state_dict = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, criterion, optimizer, train_loader, device
        )
        val_loss, val_acc = evaluate(model, criterion, val_loader, device)

        print(
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict()
            print(f"New best model with val acc {best_val_acc:.4f}")

    if best_state_dict is not None:
        state_dict_to_save = best_state_dict
        print(f"Saving best validation model to {output_model_path}")
    else:
        # Fallback for very small datasets where there is effectively no
        # validation split: just save the final model so the rest of the
        # pipeline (API + frontend) can still work.
        state_dict_to_save = model.state_dict()
        print("No validation data or best model tracked; saving final model.")

    torch.save(
        {
            "model_state_dict": state_dict_to_save,
            "class_names": class_names,
        },
        output_model_path,
    )
    print(f"Model saved to {output_model_path}")


if __name__ == "__main__":
    main()


