import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class WikiArtDataset(Dataset):
    def __init__(self, data_file, image_root, transform=None):
        super().__init__()
        self.image_root = os.path.normpath(image_root)
        self.transform = transform if transform else self.default_transform()

        self.samples = []
        self.total_samples = 0
        self.missing_count = 0

        with open(data_file, 'r') as f:
            for line in f:
                if line.strip() == '' or 'Path to image' in line:
                    continue  # 跳過標題或空行
                parts = line.strip().split(',,')
                if len(parts) != 2:
                    continue
                path, label = parts
                path = path.strip().replace('\\', '/')  # 統一成斜線

                # 移除可能多餘的資料夾前綴
                if path.startswith(self.image_root.replace('\\', '/')):
                    path = os.path.relpath(path, self.image_root)
                elif 'datasets/wikiart/' in path:
                    path = path.split('datasets/wikiart/')[-1]
                elif path.startswith('./') or path.startswith('.\\'):
                    path = path[2:]

                full_path = os.path.normpath(os.path.join(self.image_root, path))
                self.total_samples += 1
                if not os.path.exists(full_path):
                    print(f"❌ 找不到圖片：{full_path}，已跳過")
                    self.missing_count += 1
                    continue
                self.samples.append((full_path, int(label.strip())))

        print(f"✅ 可用圖片數量：{len(self.samples)} / 總共 {self.total_samples}，缺失 {self.missing_count}")

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
            print(f"⚠️ 無法載入圖片：{img_path}，錯誤：{e}")
            return None


def collate_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    return torch.utils.data.default_collate(batch)


