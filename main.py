import argparse
import os
import datetime
import torch
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from CustomDataset import CustomDataset
from models.Conv7_fc2_nopool import StarNet  # Starnet.py의 StarNet 클래스를 import합니다.

def main():
    parser = argparse.ArgumentParser(description="PyTorch StarNet Training")
    parser.add_argument("data", metavar="DIR", help="path to dataset")
    parser.add_argument("--arch", default="starnet", help="model architecture")
    parser.add_argument("--solver", default="adam", choices=["adam", "sgd"], help="solver algorithm")
    parser.add_argument("--workers", default=8, type=int, help="number of data loading workers")
    parser.add_argument("--epochs", default=300, type=int, help="number of total epochs to run")
    parser.add_argument("--start-epoch", default=0, type=int, help="manual epoch number (useful on restarts)")
    parser.add_argument("--batch-size", default=8, type=int, help="mini-batch size")
    parser.add_argument("--lr", default=0.0001, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum for sgd, alpha parameter for adam")
    parser.add_argument("--beta", default=0.999, type=float, help="beta parameter for adam")
    parser.add_argument("--weight-decay", default=4e-4, type=float, help="weight decay")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--evaluate", action="store_true", help="evaluate model on validation set")
    parser.add_argument("--pretrained", default=None, help="path to pre-trained model")
    parser.add_argument("--no-date", action="store_true", help="don't append date timestamp to folder")
    parser.add_argument("--milestones", default=[100, 150, 200], nargs="*", help="epochs at which learning rate is divided by 2")
    parser.add_argument("--save-dir", default="saved_models", help="directory to save the trained models")
    parser.add_argument("--patience", default=20, type=int, help="number of epochs to wait for improvement before stopping")
    parser.add_argument("--val-split", default=0.1, type=float, help="proportion of the dataset to include in the validation split")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 결과를 저장할 폴더 생성
    save_path = args.save_dir
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    train_writer = SummaryWriter(os.path.join(save_path, "train"))
    val_writer = SummaryWriter(os.path.join(save_path, "val"))
    
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    full_dataset = CustomDataset(root_dir=args.data, transform=train_transform)
    
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # 훈련 및 검증 데이터의 개수 출력
    print(f"Train 데이터 수: {len(train_dataset)}")
    print(f"Val 데이터 수: {len(val_dataset)}")
    
    # 훈련 시작 메시지 출력
    print("훈련 시작")

    model = StarNet().to(device)
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))

    param_groups = [
        {"params": model.parameters(), "weight_decay": args.weight_decay},
    ]

    if args.solver == "adam":
        optimizer = torch.optim.Adam(param_groups, args.lr, betas=(args.momentum, args.beta))
    elif args.solver == "sgd":
        optimizer = torch.optim.SGD(param_groups, args.lr, momentum=args.momentum)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)

    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(args.start_epoch, args.epochs):
        train_loss = train(train_loader, model, optimizer, epoch, train_writer, device, args.print_freq)
        val_loss = validate(val_loader, model, epoch, val_writer, device, args.print_freq)

        scheduler.step()

        train_writer.add_scalar("Loss/train", train_loss, epoch)
        val_writer.add_scalar("Loss/val", val_loss, epoch)

        is_best = val_loss < best_loss

        if is_best:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }, is_best, save_path)

        if epochs_no_improve >= args.patience:
            print("Validation loss did not improve for {} epochs. Stopping training.".format(args.patience))
            break


def train(train_loader, model, optimizer, epoch, writer, device, print_freq):
    model.train()
    losses = AverageMeter()

    for i, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)

        output = model(images)
        loss = F.mse_loss(output, target)

        losses.update(loss.item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % print_freq == 0:
            print(f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                  f"Loss {losses.val:.4f} ({losses.avg:.4f})")
            loss_train = losses.val
    return losses.avg

def validate(val_loader, model, epoch, writer, device, print_freq):
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = F.mse_loss(output, target)

            losses.update(loss.item(), images.size(0))
            
            if i % print_freq == 0:
                print(f"Test: [{i}/{len(val_loader)}]\t"
                      f"Loss {losses.val:.4f} ({losses.avg:.4f})")
                loss_val = losses.val
    print(f" * Loss {losses.avg:.4f}")

    return losses.avg

class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        torch.save(state, os.path.join(save_path, 'model_best.pth.tar'))

if __name__ == "__main__":
    main()
