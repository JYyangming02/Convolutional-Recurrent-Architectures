import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
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
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    # === Config ===
    image_root = "data/raw"
    train_file = "./datasets/Artist/artist_train"
    val_file = "./datasets/Artist/artist_val"
    class_file = "./datasets/Artist/artist_class"
    batch_size = 32
    num_epochs = 20
    learning_rate = 1e-4
    num_classes = len(open(class_file).readlines())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Dataset & Dataloader ===
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

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
    main()
