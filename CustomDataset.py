import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms

# Custom transformation to convert numpy arrays to tensors

class ArrayToTensor(object):
    def __call__(self, sample):
        if isinstance(sample, torch.Tensor):
            return sample.clone().detach()  # 이미 Tensor인 경우, clone().detach()를 사용
        else:
            return torch.tensor(sample, dtype=torch.float32)  # Tensor가 아닌 경우, torch.tensor 사용


# Define your image and flow transforms
# 해상도 변경 원할 시 두 transform모두 변경 후 main문 transformation = TRUE

input_transform = transforms.Compose([
    
    transforms.ToTensor(),  # Convert PIL Image to Tensor
    transforms.Resize((512, 512)),
    transforms.Normalize(mean=[0.45], std=[1])  # Normalize RGB images
])

target_transform = transforms.Compose([
    
    ArrayToTensor(),  # Convert optical flow numpy array to Tensor
    transforms.Resize((512, 512)),
    transforms.Normalize(mean=[0], std=[1])  # Normalize flow data
])


#아래 원본 코드를 참고해서 작성

# input_transform = transforms.Compose(
#         [
#             flow_transforms.ArrayToTensor(),
#             transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
#             transforms.Normalize(mean=[0.45, 0.432, 0.411], std=[1, 1, 1]),
#         ]
#     )
#     target_transform = transforms.Compose(
#         [
#             flow_transforms.ArrayToTensor(),
#             transforms.Normalize(mean=[0, 0], std=[args.div_flow, args.div_flow]),
#         ]
#     )

class CustomDataset(Dataset):
    def __init__(self, root_dir, transformation):
        self.root_dir = root_dir
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.transformation = transformation
        self.image_pairs = []
        
        for file_name in os.listdir(root_dir):
            if file_name.endswith('_Starimage_1.png'):
                base_name = file_name.replace('_Starimage_1.png', '')
                img1_path = os.path.join(root_dir, base_name + '_Starimage_1.png')
                img2_path = os.path.join(root_dir, base_name + '_Starimage_2.png')
                flo_path = os.path.join(root_dir, base_name + '_optical_flow.flo')

                if os.path.exists(img1_path) and os.path.exists(img2_path) and os.path.exists(flo_path):
                    self.image_pairs.append((img1_path, img2_path, flo_path))
                else:
                    print(f"Expected files not found for base name: {base_name}")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, flo_path = self.image_pairs[idx]

        # Load images
        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')

        # Load optical flow
        flow = self.load_optical_flow(flo_path)
        flow = flow.permute(2, 0, 1)
        # Apply transformations if any
        if self.transformation:
  
            img1 = self.input_transform(img1)
            print("img2_bef:",img2.size)
            img2 = self.input_transform(img2)
            print("img2_af:",img2.shape)
            print("flow_bef:",flow.shape)
            flow = self.target_transform(flow)
            print("flow_af:",flow.shape)
        
        return (torch.cat((img1, img2), dim=0), flow)

    def load_optical_flow(self, path):
        with open(path, 'rb') as f:
            # Read magic number
            magic = f.read(4).decode('ascii')
            if magic != 'PIEH':
                raise ValueError('Invalid .flo file: Incorrect magic number')
            
            # Read width and height
            width = np.fromfile(f, dtype=np.int32, count=1)[0]
            height = np.fromfile(f, dtype=np.int32, count=1)[0]
            
            # Read flow data
            flow_data = np.fromfile(f, dtype=np.float32).reshape((height, width, 2))
            
        # Convert to PyTorch tensor
        flow_tensor = torch.tensor(flow_data, dtype=torch.float32)
        
        return flow_tensor
