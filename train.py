<<<<<<< HEAD
import os
import matplotlib.pyplot as plt
=======
>>>>>>> bd7917a1882f75f8490a67395e2b0132353c4541
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
<<<<<<< HEAD

from models.CViT import CViT
from models.Resnet_LSTM import ResNetLSTM
from datasets.data_loader import WikiArtDataset, collate_skip_none
from utils.evaluate_metrics import evaluate_metrics

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

=======
import os

from models.CViT import CViT
from datasets.artist_dataset import WikiArtArtistDataset


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
>>>>>>> bd7917a1882f75f8490a67395e2b0132353c4541
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

<<<<<<< HEAD
    return running_loss / total, correct / total
=======
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc
>>>>>>> bd7917a1882f75f8490a67395e2b0132353c4541


def evaluate(model, dataloader, criterion, device):
    model.eval()
<<<<<<< HEAD
    running_loss, correct, total = 0.0, 0, 0
=======
    running_loss = 0.0
    correct = 0
    total = 0
>>>>>>> bd7917a1882f75f8490a67395e2b0132353c4541

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

<<<<<<< HEAD
    return running_loss / total, correct / total


def plot_metrics(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Plot Loss & Accuracy ---
    axs[0].plot(epochs, history["train_loss"], label='Train Loss', color='blue', linestyle='-')
    axs[0].plot(epochs, history["val_loss"], label='Val Loss', color='blue', linestyle='--')
    axs[0].plot(epochs, history["train_acc"], label='Train Acc', color='green', linestyle='-')
    axs[0].plot(epochs, history["val_acc"], label='Val Acc', color='green', linestyle='--')
    axs[0].set_ylabel("Loss / Accuracy")
    axs[0].legend()
    axs[0].set_title("Loss and Accuracy")

    # --- Plot Learning Rate ---
    axs[1].plot(epochs, history["lr"], label='Learning Rate', color='red')
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Learning Rate")
    axs[1].legend()
    axs[1].set_title("Learning Rate Schedule")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
=======
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc
>>>>>>> bd7917a1882f75f8490a67395e2b0132353c4541


def main():
    # === Config ===
<<<<<<< HEAD
    image_root = "./datasets/wikiart"
    train_file = "./datasets/Artist/artist_train"
    val_file = "./datasets/Artist/artist_val"
    class_file = "./datasets/Artist/artist_class"

    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    patience = 5  # Early stopping patience
    save_dir = "checkpoints"

    num_classes = len(open(class_file).readlines())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using:{device}")
    os.makedirs(save_dir, exist_ok=True)
=======
    image_root = "data/raw"
    train_file = "./datasets/Artist/artist_train"
    val_file = "./datasets/Artist/artist_val"
    class_file = "./datasets/Artist/artist_class"
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-4
    num_classes = len(open(class_file).readlines())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
>>>>>>> bd7917a1882f75f8490a67395e2b0132353c4541

    # === Dataset & Dataloader ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
<<<<<<< HEAD
    train_dataset = WikiArtDataset(train_file, image_root, transform)
    val_dataset = WikiArtDataset(val_file, image_root, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_skip_none)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_skip_none)

    # === Model ===
    model = ResNetLSTM(num_classes=num_classes).to(device)

    # === Loss, Optimizer ===
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # === History Tracking ===
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}

    # === Early Stopping Setup ===
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"\n[Epoch {epoch + 1}/{num_epochs}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Save stats
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(optimizer.param_groups[0]['lr'])

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            print("Saved best model.")
        else:
            epochs_no_improve += 1
            print(f"No improvement. ({epochs_no_improve}/{patience})")

        # Early stopping check
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(save_dir, f"resnet_lstm_epoch{epoch + 1}.pt"))

    # === Plot metrics ===
    plot_metrics(history, os.path.join(save_dir, "combined_metrics.png"))

    print("\nEvaluating best model with full metrics:")
    class_names = [line.strip() for line in open(class_file)]
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt")))
    evaluate_metrics(model, val_loader, class_names, device)


if __name__ == "__main__":
=======

    train_dataset = WikiArtArtistDataset(train_file, class_file, image_root, transform)
    val_dataset = WikiArtArtistDataset(val_file, class_file, image_root, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # === Model ===
    model = CViT(num_classes=num_classes)
    model.to(device)

    # === Loss & Optimizer ===
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # === Training Loop ===
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(f"[Epoch {epoch+1}/{num_epochs}]",
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} ||",
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # save checkpoint
        torch.save(model.state_dict(), f"checkpoints/cvit_epoch{epoch+1}.pt")


if __name__ == '__main__':
    os.makedirs("checkpoints", exist_ok=True)
>>>>>>> bd7917a1882f75f8490a67395e2b0132353c4541
    main()
