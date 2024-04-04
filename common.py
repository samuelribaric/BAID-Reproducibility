'''
Common utils for training
'''

import os
import torch
from config import BASE_PATH, SAVE_DIR

import os
import torch
from config import BASE_PATH, SAVE_DIR

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 10)) if epoch < 40 else args.lr * (0.1 ** 4)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(args, model, optimizer, epoch):
    checkpoint_dir = '/kaggle/working' if BASE_PATH.startswith('/kaggle/input') else args.checkpoint_dir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    save_path = os.path.join(checkpoint_dir, 'model_best.pth')
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, save_path)

def load_checkpoint(args, model, optimizer):
    checkpoint_path = os.path.join(args.checkpoint_dir, 'model_best.pth')
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    return start_epoch, optimizer
