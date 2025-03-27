import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from datasets.data_loader import WikiArtDataset, collate_skip_none
from models.CViT import CViT
import matplotlib.pyplot as plt
from PIL import Image

# === CONFIG ===
task = "genre"  # "artist", "style", or "genre"
image_root = "./datasets/wikiart"
train_file = f"./datasets/{task}_train"
val_file = f"./datasets/{task}_val"
class_file = f"./datasets/{task}_class"

model_path = f"checkpoints/{task}_best_model.pt"
outlier_dir = "outliers"

os.makedirs(outlier_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(open(class_file).readlines())

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# === Load data and model ===
dataset = WikiArtDataset(val_file, image_root, transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_skip_none)

model = CViT(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# === Analyze uncertainty ===
entropy_list = []
softmax = torch.nn.Softmax(dim=1)

with torch.no_grad():
    for idx, batch in enumerate(tqdm(dataloader, desc="Analyzing")):
        if batch is None:
            continue
        image, label = batch
        image = image.to(device)
        output = model(image)
        prob = softmax(output)

        entropy = -torch.sum(prob * torch.log(prob + 1e-9), dim=1).item()
        predicted = prob.argmax(dim=1).item()
        true_label = label.item()

        entropy_list.append((entropy, predicted, true_label, dataset.samples[idx][0]))

# === Sort by highest entropy (most uncertain) ===
entropy_list.sort(reverse=True, key=lambda x: x[0])

# === Save top-N outliers ===
N = 20
print("\nTop 20 most uncertain predictions (possible outliers):")
for i in range(min(N, len(entropy_list))):
    ent, pred, true, path = entropy_list[i]
    print(f"[{i+1}] Entropy: {ent:.4f} | True: {true} | Pred: {pred} | {path}")
    # Save image with title
    try:
        img = Image.open(path).convert("RGB")
        plt.imshow(img)
        plt.title(f"Entropy: {ent:.2f}\nTrue: {true} | Pred: {pred}")
        plt.axis("off")
        plt.savefig(os.path.join(outlier_dir, f"outlier_{i+1}.png"))
        plt.close()
    except Exception as e:
        print(f"Failed to save image: {path}, error: {e}")
