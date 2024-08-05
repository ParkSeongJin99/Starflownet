import os
import torch
import torch.nn.functional as F
import torch.optim
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from CustomDataset_val import CustomDataset  # CustomDataset.py의 CustomDataset 클래스를 import합니다.
from models.Starflow import Starflow  # Starflow.py의 Starflow 클래스를 import합니다.
import struct

def visualize_star_movement(first_image, second_image, predicted_flow, target_flow, step=1):
    """
    이미지 위에 Optical Flow 벡터를 시각화합니다.

    Args:
    - image (Tensor): 시각화할 이미지 텐서, shape: (1, H, W)
    - predicted_flow (Tensor): 예측된 Optical Flow 텐서, shape: (2, H, W)
    - target_flow (Tensor): 실제 Optical Flow 텐서, shape: (2, H, W)
    - step (int): 화살표의 밀도를 조절하기 위한 스텝 크기
    """
    # 이미지 및 Flow 텐서를 numpy 배열로 변환
    first_image = first_image.squeeze().numpy()
    second_image = second_image.squeeze().numpy()
    predicted_flow = predicted_flow.numpy()
    target_flow = target_flow.numpy()

    H, W = first_image.shape
    Y, X = np.mgrid[0:H:step, 0:W:step]

    pred_U = predicted_flow[0]
    pred_V = predicted_flow[1]
    mask_pred = first_image > 0
    pred_U_mask = pred_U[mask_pred]
    pred_V_mask = pred_V[mask_pred]
    Y_pred, X_pred = np.mgrid[0:H, 0:W]
    Y_pred_mask  = Y_pred[mask_pred]
    X_pred_mask  = X_pred[mask_pred]
    pred_U = predicted_flow[0, ::step, ::step]
    pred_V = predicted_flow[1, ::step, ::step]
    Y_pred = Y_pred[::step, ::step]
    X_pred = X_pred[::step, ::step]
    # Filter out zero vectors for target flow
    target_U = target_flow[0]
    target_V = target_flow[1]
    mask_target = np.sqrt(target_U**2 + target_V**2) > 0
    
    target_U = target_U[mask_target]
    target_V = target_V[mask_target]
    
    Y_target, X_target = np.mgrid[0:H, 0:W]
    Y_target = Y_target[mask_target]
    X_target = X_target[mask_target]

    # 시각화
    plt.figure(figsize=(10, 10))
    plt.imshow(first_image, cmap='gray')
    plt.imshow(second_image, cmap='gray')
    plt.quiver(X_pred, Y_pred, pred_U, pred_V, color='r', scale=1, scale_units='xy', angles='xy', headwidth=3, headlength=5, label='Predicted')
    plt.quiver(X_pred_mask, Y_pred_mask, pred_U_mask, pred_V_mask, color='r', scale=1, scale_units='xy', angles='xy', headwidth=3, headlength=5, label='Predicted')
    plt.quiver(X_target, Y_target, target_U, target_V, color='b', scale=1, scale_units='xy', angles='xy', headwidth=3, headlength=5, label='Target')
    plt.title('Star Movement Visualization')
    plt.legend()
    plt.show()

def save_flow_to_bin(flow, file_path):
    """
    Optical Flow 데이터를 바이너리 파일로 저장합니다.

    Args:
    - flow (Tensor): 저장할 Optical Flow 텐서, shape: (2, H, W)
    - file_path (str): 저장할 파일의 경로
    """
    flow = flow.numpy()  # Convert Tensor to numpy array
    H, W = flow.shape[1:]  # Extract height and width

    # Reshape the flow array to (H, W * 2)
    flow_data = np.reshape(flow, (H, W * 2))

    # Write to binary file
    with open(file_path, 'wb') as f:
        # Write the magic number 'PIEH'
        f.write(b'PIEH')

        # Write width and height
        f.write(struct.pack('i', W))
        f.write(struct.pack('i', H))

        # Write flow data as float32
        flow_data.astype('float32').tofile(f)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Starflow Validation")
    parser.add_argument("model", help="model_name")
    parser.add_argument("--data", metavar="DIR", default="dataset", help="path to dataset")
    parser.add_argument("--modelpath", help="directory to load the trained models")
    parser.add_argument("--output", help="directory to save the predicted flow")

    args = parser.parse_args()

    # 데이터셋을 읽어오는 DataLoader를 설정합니다.
    dataset = CustomDataset(root_dir=args.data, transformation=1)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 사전 훈련된 모델을 로드합니다.
    pretrained_model_path = os.path.join(args.modelpath, args.model)  # 사전 훈련된 모델의 경로를 지정합니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Starflow().to(device)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu'))['state_dict'])
    model.eval()

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    for sample_data in data_loader:
        input_images, target_flow, filename = sample_data
        filename = filename[0]  # Batch size가 1이므로, 첫 번째 파일 이름만 사용

        with torch.no_grad():
            output_flow = model(input_images.to(device)).cpu()

        h, w = target_flow.size()[-2:]
        output_flow = F.interpolate(output_flow, (h, w))

        first_image = input_images[0][0]  # 첫 번째 이미지 (입력 이미지)
        second_image = input_images[0][1]  # 두 번째 이미지 (입력 이미지)
        predicted_first_flow = output_flow[0]  # 예측된 첫 번째 flow
        target_first_flow = target_flow[0]  # 실제 첫 번째 flow

        # 시각화
        visualize_star_movement(first_image, second_image, predicted_first_flow, target_first_flow, 15)

        # 예측된 flow를 텍스트 파일로 저장
        flow_filename = os.path.join(args.output, f'{filename}_flow.bin')
        save_flow_to_bin(predicted_first_flow, flow_filename)
        print(f"Predicted flow saved to {flow_filename}")

if __name__ == "__main__":
    main()
