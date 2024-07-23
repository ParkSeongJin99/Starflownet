import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = []
        self.labels = []

        image_dir = os.path.join(root_dir, 'images')
        label_dir = os.path.join(root_dir, 'labels')

        for label_file in os.listdir(label_dir):
            label_path = os.path.join(label_dir, label_file)
            try:
                with open(label_path, 'r') as f:
                    label = float(f.readline().strip())
            except Exception as e:
                print(f"Error loading label file: {label_path}. Error: {e}")
                continue

            image_base_name = label_file.replace('.txt', '')
            img1_path = os.path.join(image_dir, image_base_name + '_1.png')
            img2_path = os.path.join(image_dir, image_base_name + '_2.png')

            if os.path.exists(img1_path) and os.path.exists(img2_path):
                self.image_pairs.append((img1_path, img2_path))
                self.labels.append(label)
            else:
                print(f"Image files not found for label: {label_file}. Expected paths: '{img1_path}', '{img2_path}'")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        label = self.labels[idx]

        try:
            img1 = Image.open(img1_path).convert('L')
            img2 = Image.open(img2_path).convert('L')
        except Exception as e:
            print(f"Error loading images at paths: '{img1_path}', '{img2_path}'. Error: {e}")
            raise

        if self.transform:
            try:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            except Exception as e:
                print(f"Error applying transform to images at paths: '{img1_path}', '{img2_path}'. Error: {e}")
                raise

        try:
            image_pair = torch.cat((img1, img2), dim=0)
        except Exception as e:
            print(f"Error concatenating images at paths: '{img1_path}', '{img2_path}'. Error: {e}")
            raise

        try:
            label = torch.tensor([label], dtype=torch.float32)
        except Exception as e:
            print(f"Error converting label to tensor. Label: {label}. Error: {e}")
            raise

        return image_pair, label
