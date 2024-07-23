import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from CustomDataset import CustomDataset  # CustomDataset.py의 CustomDataset 클래스를 import합니다.
from models.Conv6_fc2_nopool import StarNet  # Starnet.py의 StarNet 클래스를 import합니다.

def main():
    parser = argparse.ArgumentParser(description="PyTorch StarNet Validation")
    
    parser.add_argument("model", help="model_name")
    parser.add_argument("--datapath", metavar="DIR", default="Validation", help="path to dataset")
    parser.add_argument("--txtpath", metavar="DIR", default="Validation", help="path to dataset")
    parser.add_argument("--output", metavar="FILE", default="output.txt", help="path to output text file")
    parser.add_argument("--modelpath", help="directory to load the trained models")
    args = parser.parse_args()

    # 이미지 데이터셋이 있는 폴더 경로를 지정합니다.
    data_dir = args.datapath
    # 저장할 txt파일 이름을 모델 이름으로 변경합니다
    args.output = os.path.join(args.txtpath,args.model+'.txt')

    

    # 데이터셋을 읽어오는 DataLoader를 설정합니다.
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset = CustomDataset(root_dir=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 사전 훈련된 모델을 로드합니다.
    pretrained_model_path = os.path.join(args.modelpath,args.model)  # 사전 훈련된 모델의 경로를 지정합니다.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StarNet().to(device)
    model.load_state_dict(torch.load(pretrained_model_path, map_location=torch.device('cpu'))['state_dict'])
    model.eval()
    
    # 각 이미지에 대한 결과값을 계산하고 출력합니다.
    all_outputs = []
    all_labels = []
    all_percent_errors = []

    # 출력 파일 경로를 절대 경로로 변경
    output_file_path = os.path.abspath(args.output)
    print(output_file_path)
    with open(output_file_path, 'w') as f:
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            output = outputs.item()
            label = labels.item()
            error = output - label
            percent_error = (error / label) * 100 if label != 0 else float('inf')
            
            all_outputs.append(output)
            all_labels.append(label)
            all_percent_errors.append(percent_error)
            
            # 출력값과 레이블을 파일에 저장
            f.write(f"{label} {output}\n")
            
            print(f"Output: {output:.2f} deg/s, Label: {label:.2f} deg/s, Error: {error:.2f} deg/s, Percent Error: {percent_error:.2f}%")
    
    # 평균 절대 오차 (MAE) 및 루트 평균 제곱 오차 (RMSE) 계산
    mae = np.mean(np.abs(np.array(all_outputs) - np.array(all_labels)))
    rmse = np.sqrt(np.mean((np.array(all_outputs) - np.array(all_labels)) ** 2))
    
    # 결과를 표로 출력
    print("\nEvaluation Metrics:")
    print(f"{'Metric':<20} {'Value':<20}")
    print(f"{'-'*40}")
    print(f"{'Mean Absolute Error (MAE)':<20} {mae:<20.4f} deg/s")
    print(f"{'Root Mean Squared Error (RMSE)':<20} {rmse:<20.4f} deg/s")
    
    # 회귀 그래프를 그립니다.
    plt.figure(figsize=(12, 6))
    
    # 첫 번째 그래프: 실제 레이블과 예측값의 회귀 그래프
    plt.subplot(1, 2, 1)
    plt.scatter(all_labels, all_outputs, color='blue', label='Predicted vs Actual')
    plt.plot([min(all_labels), max(all_labels)], [min(all_labels), max(all_labels)], color='red', linestyle='--', label='Ideal')
    plt.xlabel('Actual Labels (deg/s)')
    plt.ylabel('Predicted Outputs (deg/s)')
    plt.title('Regression Graph')
    plt.legend()
    plt.grid(True)
    
    # 두 번째 그래프: 실제 레이블과 퍼센트 에러 그래프
    plt.subplot(1, 2, 2)
    plt.scatter(all_labels, all_percent_errors, color='green', label='Percent Error vs Actual')
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Error Line')
    plt.xlabel('Actual Labels (deg/s)')
    plt.ylabel('Percent Error (%)')
    plt.title('Percent Error Graph')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle('Angular Velocity Prediction Analysis')
    plt.show()

if __name__ == "__main__":
    main()
