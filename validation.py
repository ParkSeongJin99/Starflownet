import os
import torch
import torch.nn.functional as F
import torch.optim
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from CustomDataset import CustomDataset  # CustomDataset.py의 CustomDataset 클래스를 import합니다.
from models.Starflow import Starflow  # Starflow.py의 Starflow 클래스를 import합니다.

def visualize_star_movement(image, predicted_flow, target_flow):
    """
    이미지 위에 Optical Flow 벡터를 시각화합니다.

    Args:
    - image (Tensor): 시각화할 이미지 텐서, shape: (1, H, W)
    - predicted_flow (Tensor): 예측된 Optical Flow 텐서, shape: (2, H, W)
    - target_flow (Tensor): 실제 Optical Flow 텐서, shape: (2, H, W)
    """
    # 이미지 및 Flow 텐서를 numpy 배열로 변환
    image = image.squeeze().numpy()
    predicted_flow = predicted_flow.numpy()
    target_flow = target_flow.numpy()

    H, W = image.shape
    Y, X = np.mgrid[0:H, 0:W]

    pred_U = predicted_flow[0]
    pred_V = predicted_flow[1]

    target_U = target_flow[0]
    target_V = target_flow[1]

    # 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.quiver(X, Y, pred_U, pred_V, color='r', scale=1, scale_units='xy', angles='xy', headwidth=3, headlength=5, label='Predicted')
    plt.quiver(X, Y, target_U, target_V, color='b', scale=1, scale_units='xy', angles='xy', headwidth=3, headlength=5, label='Target')
    plt.title('Star Movement Visualization')
    plt.legend()
    plt.show()

def visualize_star_movement_1(image, target_flow, step=16):
    """
    이미지 위에 Optical Flow 벡터를 시각화합니다.

    Args:
    - image (Tensor): 시각화할 이미지 텐서, shape: (1, H, W)
    - predicted_flow (Tensor): 예측된 Optical Flow 텐서, shape: (2, H, W)
    - target_flow (Tensor): 실제 Optical Flow 텐서, shape: (2, H, W)
    - step (int): 화살표의 밀도를 조절하기 위한 스텝 크기
    """
    # 이미지 및 Flow 텐서를 numpy 배열로 변환
    image = image.squeeze().numpy()
   
    target_flow = target_flow.numpy()

    H, W = image.shape
    Y, X = np.mgrid[0:H:step, 0:W:step]

    # Filter out zero vectors for target flow
    target_U = target_flow[0]
    target_V = target_flow[1]
    mask = np.sqrt(target_U**2 + target_V**2) > 0

    target_U = target_U[mask]
    target_V = target_V[mask]
    Y_target, X_target = np.mgrid[0:H, 0:W]
    Y_target = Y_target[mask]
    X_target = X_target[mask]

    # 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.quiver(X_target, Y_target, target_U, target_V, color='b', scale=1, scale_units='xy', angles='xy', headwidth=3, headlength=5, label='Target')
    plt.title('Star Movement Visualization')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Starflow Validation")
    parser.add_argument("model", help="model_name")
    parser.add_argument("--data", metavar="DIR",default="dataset", help="path to dataset")
    parser.add_argument("--modelpath", help="directory to load the trained models")

    args = parser.parse_args()

    # 이미지 데이터셋이 있는 폴더 경로를 지정합니다.
    #data_dir = args.datapath
    # 저장할 txt파일 이름을 모델 이름으로 변경합니다
    #args.output = os.path.join(args.txtpath,args.model+'.txt')

    

    # 데이터셋을 읽어오는 DataLoader를 설정합니다.

    
    dataset = CustomDataset(root_dir=args.data, transformation=1)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 사전 훈련된 모델을 로드합니다.
    pretrained_model_path = os.path.join(args.modelpath,args.model)  # 사전 훈련된 모델의 경로를 지정합니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Starflow().to(device)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu'))['state_dict'])
    model.eval()

    
    sample_data = next(iter(data_loader))
    
    input_images, target_flow = sample_data
    model.eval()
    with torch.no_grad():
        output_flow = model(input_images.to(device)).cpu()
    
    
    # Since Target pooling is not very precise when sparse,
    # take the highest resolution prediction and upsample it instead of downsampling target
    h, w = target_flow.size()[-2:]
    output_flow = F.interpolate(output_flow, (h, w))

    first_image = input_images[0][:1]  # 첫 번째 이미지 (입력 이미지)
    print(first_image.shape)
    predicted_first_flow = output_flow[0]  # 예측된 첫 번째 flow
    print(predicted_first_flow.shape)
    target_first_flow = target_flow[0]  # 실제 첫 번째 flow
    print(target_first_flow.shape)

    visualize_star_movement_1(first_image, target_first_flow)
    
    
   
if __name__ == "__main__":
    main()
