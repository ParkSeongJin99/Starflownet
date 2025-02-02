import os
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

# Custom transformation to convert numpy arrays to tensors
class ArrayToTensor(object):
    def __call__(self, sample):
        if isinstance(sample, torch.Tensor):
            return sample.clone().detach()  # 이미 Tensor인 경우, clone().detach()를 사용
        else:
            return torch.tensor(sample, dtype=torch.float32)  # Tensor가 아닌 경우, torch.tensor 사용

# Define your image and flow transforms
input_transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to Tensor
    transforms.Resize((512, 512)),
    transforms.Normalize(mean=[0.45], std=[1])  # Normalize gray scale images
])

def resize_and_scale_optical_flow(flow, new_size):
    original_size = (flow.shape[2], flow.shape[1])  # (width, height)
    scale_x = new_size[1] / original_size[0]
    scale_y = new_size[0] / original_size[1]

    # Resize the flow field using bilinear interpolation
    resized_flow = F.interpolate(flow.unsqueeze(0), size=new_size, mode='bilinear', align_corners=False).squeeze(0)

    # Scale the flow vectors
    resized_flow[0] *= scale_x  # Scale x direction
    resized_flow[1] *= scale_y  # Scale y direction

    return resized_flow

# Define the transformation for optical flow data
target_transform = transforms.Compose([
    ArrayToTensor(),  # Convert optical flow numpy array to Tensor
    transforms.Lambda(lambda flow: resize_and_scale_optical_flow(flow, (512, 512))),
    transforms.Normalize(mean=[0], std=[1])  # Normalize flow data
])

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
                    self.image_pairs.append((img1_path, img2_path, flo_path, base_name))
                else:
                    print(f"Expected files not found for base name: {base_name}")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, flo_path, base_name = self.image_pairs[idx]

        try:
            # Load images
            img1 = Image.open(img1_path).convert('L')
            img2 = Image.open(img2_path).convert('L')
        except UnidentifiedImageError as e:
            print(f"Skipping corrupted image: {img1_path} or {img2_path}. Error: {e}")
            return self.__getitem__((idx + 1) % len(self.image_pairs))

        # Load optical flow
        flow = self.load_optical_flow(flo_path)
        flow = flow.permute(2, 0, 1)

        # Apply transformations if any
        if self.transformation:
            img1 = self.input_transform(img1)
            img2 = self.input_transform(img2)
            flow = self.target_transform(flow)

        return (torch.cat((img1, img2), dim=0), flow, base_name)

    def load_optical_flow(self, path):
        with open(path, 'rb') as f:
            # Read magic number
            magic = f.read(4).decode('ascii')
            if magic != 'PIEH':
                raise ValueError('Invalid .flo file: Incorrect magic number')
            
            # Read width and height
            width = np.fromfile(f, dtype=np.int32, count=1)[0]
            height = np.fromfile(f, dtype=np.int32, count=1)[0]
            
            # Total number of elements for x and y components
            total_elements = width * height
            
            # Read flow data
            flow_data = np.fromfile(f, dtype=np.float32, count=total_elements * 2)
            
            # Separate into x and y components, reading height rows at a time
            x = flow_data[:total_elements].reshape((width, height)).transpose()
            y = flow_data[total_elements:].reshape((width, height)).transpose()
            
            # Stack x and y components into a flow tensor
            flow = np.stack((x, y), axis=-1)
            
        # Convert to PyTorch tensor
        flow_tensor = torch.tensor(flow, dtype=torch.float32)
        
        return flow_tensor
