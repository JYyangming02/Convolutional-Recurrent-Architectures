import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class WikiArtArtistDataset(Dataset):
    def __init__(self, data_file, class_file, image_root, transform=None):
        super().__init__()
        self.image_root = image_root
        self.transform = transform if transform else self.default_transform()

        # 讀取類別對應表
        with open(class_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # 讀取圖片與標籤路徑
        self.samples = []
        with open(data_file, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                full_path = os.path.join(self.image_root, path)
                self.samples.append((full_path, self.class_to_idx[label]))

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
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label
