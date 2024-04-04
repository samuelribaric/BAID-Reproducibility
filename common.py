'''
Common utils for training
'''

import os
import torch
from config import BASE_PATH, SAVE_DIR

def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    if epoch < 40:
        lr = args.lr * (0.1 ** (epoch // 10))
    else:
        lr = args.lr * (0.1 ** 4)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(args, model, epoch):
    # Determine the base path for saving based on the environment
    if BASE_PATH.startswith('/kaggle/input'):
        # In Kaggle, adjust to a directory where you have write access
        checkpoint_dir = '/kaggle/working'
    else:
        # Locally, use the directory specified in args
        checkpoint_dir = args.checkpoint_dir
    
    # Ensure the directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    # Construct the save path and save the model checkpoint
    save_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_path)
