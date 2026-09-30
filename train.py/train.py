"""Train a ResNet-18 transfer-learning baseline on folder-based image data."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(data_dir, batch_size, seed):
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val = datasets.ImageFolder(data_dir / "val", transform=eval_tf)
    if train.classes != val.classes:
        raise ValueError("Train and val folders must contain the same class names.")
    if not len(train) or not len(val):
        raise ValueError("Both train and val folders must contain images.")
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    return train.classes, train_loader, val_loader


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = count = 0
    all_labels, all_predictions = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        predictions = logits.argmax(dim=1)
        n = labels.size(0)
        total_loss += loss.item() * n
        correct += (predictions == labels).sum().item()
        count += n
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
    return total_loss / count, correct / count, all_labels, all_predictions


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("--epochs and --batch-size must be positive.")
    set_seed(args.seed)
    classes, train_loader, val_loader = make_loaders(args.data_dir, args.batch_size, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.learning_rate
    )
    loss_fn = nn.CrossEntropyLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy, best_labels, best_predictions = -1.0, [], []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc, labels, predictions = run_epoch(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f}, acc {train_acc:.3f} | val loss {val_loss:.4f}, acc {val_acc:.3f}")
        if val_acc > best_accuracy:
            best_accuracy, best_labels, best_predictions = val_acc, labels, predictions
            torch.save({"model_state_dict": model.state_dict(), "classes": classes}, args.output_dir / "best_model.pt")
    report = classification_report(best_labels, best_predictions, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(best_labels, best_predictions, labels=list(range(len(classes)))).tolist()
    results = {"best_validation_accuracy": best_accuracy, "classes": classes,
               "classification_report": report, "confusion_matrix": matrix,
               "confusion_matrix_labels": classes}
    (args.output_dir / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved best model and evaluation outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
