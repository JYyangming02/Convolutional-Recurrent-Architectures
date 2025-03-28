import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ranger21 import Ranger21

from models.Resnet_LSTM import ResNetLSTM
from datasets.csv_data_loader import csv_Dataset, collate_skip_none
from utils.evaluate_metrics import evaluate_metrics

def load_pretrained_exclude_classifier(model, checkpoint_path, device):
    print(f"\nLoading pretrained weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model_dict = model.state_dict()

    # Filter out classifier layers with size mismatch
    filtered_dict = {
        k: v for k, v in state_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }

    # Show how many layers successfully loaded
    print(f"Loaded {len(filtered_dict)}/{len(model_dict)} layers (excluding classifier)")

    # Update the model weights
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        if batch is None: continue
        images, labels = batch
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if batch is None: continue
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total

def plot_metrics(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 8))

    # Plot Loss
    plt.plot(epochs, history["train_loss"], label='Train Loss', color='#1f77b4', linestyle='-')
    plt.plot(epochs, history["val_loss"], label='Val Loss', color='#1f77b4', linestyle='--')

    # Plot Accuracy
    plt.plot(epochs, history["train_acc"], label='Train Acc', color='#2ca02c', linestyle='-')
    plt.plot(epochs, history["val_acc"], label='Val Acc', color='#2ca02c', linestyle='--')

    # Plot Scaled LR
    max_loss = max(history["train_loss"] + history["val_loss"])
    max_lr = max(history["lr"])
    scaled_lr = [lr * (max_loss / max_lr) * 0.3 for lr in history["lr"]]
    plt.plot(epochs, scaled_lr, label='Learning Rate (scaled)', color='#d62728', linestyle='-', alpha=0.6)

    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy / Scaled LR")
    plt.title("Training & Validation Metrics")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    task = "style" # genre, style
    file = "Style" # Genre, Style
    image_root = "./datasets/wikiart"
    train_file = f"./datasets/{file}/{task}_train.csv"
    val_file = f"./datasets/{file}/{task}_val.csv"
    class_file = f"./datasets/{file}/{task}_class"
    pretrained_model_path = "checkpoints/artist_best_model.pt"

    batch_size = 32
    num_epochs = 30
    learning_rate = 1e-4
    patience = 3
    save_dir = "checkpoints"

    num_classes = len(open(class_file, encoding='utf-8').readlines())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = csv_Dataset(train_file, image_root, transform)
    val_dataset = csv_Dataset(val_file, image_root, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_skip_none)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_skip_none)

    # === Model and load pretrained (excluding classifier) ===
    model = ResNetLSTM(num_classes=num_classes).to(device)
    model = load_pretrained_exclude_classifier(model, pretrained_model_path, device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = Ranger21(model.parameters(), lr=learning_rate, num_epochs=num_epochs)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch + 1}/{num_epochs}]")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]['lr'])

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, f"{task}_best_model.pt"))
            print("Saved best model.")
        else:
            epochs_no_improve += 1
            print(f"No improvement. ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    plot_metrics(history, os.path.join(save_dir, f"{task}_combined_metrics.png"))

    print("\nEvaluating best model with full metrics:")
    class_names = [line.strip() for line in open(class_file, encoding='utf-8')]
    model.load_state_dict(torch.load(os.path.join(save_dir, f"{task}_best_model.pt")))
    evaluate_metrics(model, val_loader, class_names, device)

if __name__ == "__main__":
    main()
