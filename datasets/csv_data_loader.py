import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class csv_Dataset(Dataset):
    def __init__(self, data_file, image_root, transform=None):
        super().__init__()
        self.image_root = os.path.normpath(image_root)
        self.transform = transform if transform else self.default_transform()

        self.samples = []
        self.total_samples = 0
        self.missing_count = 0

        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() == '' or 'Path to image' in line:
                    continue  # skip header or empty lines
                parts = line.strip().split(',')
                if len(parts) != 2:
                    continue
                path, label = parts
                path = path.strip().replace('\\', '/')

                # Remove redundant prefix if exists
                if path.startswith(self.image_root.replace('\\', '/')):
                    path = os.path.relpath(path, self.image_root)
                elif 'datasets/wikiart/' in path:
                    path = path.split('datasets/wikiart/')[-1]
                elif path.startswith('./') or path.startswith('.\\'):
                    path = path[2:]

                full_path = os.path.normpath(os.path.join(self.image_root, path))
                self.total_samples += 1
                if not os.path.exists(full_path):
                    print(f"Missing image: {full_path}, skipped.")
                    self.missing_count += 1
                    continue
                self.samples.append((full_path, int(label.strip())))

        print(f"Available images: {len(self.samples)} / Total: {self.total_samples}, Missing: {self.missing_count}")

    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Failed to load image: {img_path}, Error: {e}")
            return None


def collate_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)
