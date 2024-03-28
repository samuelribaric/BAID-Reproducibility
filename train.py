import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import BBDataset
from torch.utils.data import DataLoader
from models.model import SAAN
import torch.optim as optim
from common import *
import argparse
from config import BASE_PATH, SAVE_DIR
import os

train_dataset = BBDataset(file_dir='dataset', type='train', test=False)
val_dataset = BBDataset(file_dir='dataset', type='validation', test=True)

def parse_args():
    parser = argparse.ArgumentParser()
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument('--device', type=str, default=default_device)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--checkpoint_dir', type=str,
                        default=os.path.join(BASE_PATH, 'checkpoint/BAID'))
    parser.add_argument('--val_freq', type=int,
                        default=2)
    parser.add_argument('--save_freq', type=int,
                        default=2)

    return parser.parse_args()


def validate(args, model, val_loader, epoch):
    model.eval()
    device = args.device

    loss = nn.MSELoss()
    val_loss = 0.0
    with torch.no_grad():
        for step, val_data in enumerate(val_loader):
            image = val_data[0].to(device)
            label = val_data[1].to(device).float()

            predicted_label = model(image).squeeze()
            val_loss += loss(predicted_label, label).item()

    val_loss /= len(val_loader)
    print("Epoch: %3d Validation loss: %.8f" % (epoch, val_loss))


def train(args):
    device = args.device

    model = SAAN(num_classes=1)
    for name, param in model.named_parameters():
        if 'GenAes' in name:
            param.requires_grad = False
    model = model.to(device)

    loss = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=5e-4)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)

    for epoch in range(args.epoch):
        model.train()
        epoch_loss = 0.0

        for step, train_data in enumerate(train_loader):
            optimizer.zero_grad()
            image = train_data[0].to(device)
            label = train_data[1].to(device).float()

            predicted_label = model(image).squeeze()
            train_loss = loss(predicted_label, label)

            train_loss.backward()
            optimizer.step()

            epoch_loss += train_loss.item()

            print("Epoch: %3d Step: %5d / %5d Train loss: %.8f" % (epoch, step, len(train_loader), train_loss.item()))

        adjust_learning_rate(args, optimizer, epoch)

        if (epoch + 1) % args.val_freq == 0:
            validate(args, model, val_loader, epoch)

        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint(args, model, epoch)


if __name__ == '__main__':
    args = parse_args()
    train(args)
