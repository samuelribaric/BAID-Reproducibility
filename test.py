import os
import scipy.stats
from data import BBDataset
from torch.utils.data import DataLoader
from models.model import SAAN
import torch.nn as nn
import torch
import torch.optim as optim
from common import *
import argparse
import torch.nn.functional as F
import pandas as pd
import scipy
from tqdm import tqdm
from config import BASE_PATH, RESULT_DIR
import os

test_dataset = BBDataset(file_dir='dataset', type='test', test=True)

def parse_args():
    parser = argparse.ArgumentParser()
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument('--device', type=str, default=default_device)
    parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(BASE_PATH, 'checkpoint/BAID'))
    parser.add_argument('--checkpoint_name', type=str, default='model_best.pth')
    parser.add_argument('--save_dir', type=str, default=os.path.join(RESULT_DIR, 'result.csv'))
    args = parser.parse_args()

    # Adjusting save_dir for Kaggle outputs
    if BASE_PATH.startswith('/kaggle/input'):
        args.save_dir = '/kaggle/working/result'
    else:
        args.save_dir = os.path.join(BASE_PATH, 'result')

    return args




def test(args):
    device = torch.device(args.device)  # Ensure device is set correctly as a torch.device object
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    df = pd.read_csv(os.path.join(BASE_PATH,'dataset/test_set.csv'))
    predictions = []

    model = SAAN(num_classes=1)
    model = model.to(device)
    # Modify this line to include map_location, using the device variable
    
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # This line is for directly loading model checkpoints without the encapsulating dictionary
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print("No checkpoint found at {}".format(checkpoint_path))
        
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=8)
    print(f"Number of items in test_loader: {len(test_loader.dataset)}")
    with torch.no_grad():
        for step, test_data in tqdm(enumerate(test_loader)):
            image = test_data[0].to(device)

            predicted_label = model(image)
            prediction = predicted_label.squeeze().cpu().numpy()
            predictions.append(prediction * 10)
        

    scores = df['score'].values.tolist()
    print(f"Number of scores: {len(scores)}")
    print(scipy.stats.spearmanr(scores, predictions))
    print(scipy.stats.pearsonr(scores, predictions))

    acc = 0
    for i in range(len(scores)):
        cls1 = 1 if scores[i] > 5 else 0
        cls2 = 1 if predictions[i] > 5 else 0
        if cls1 == cls2:
            acc += 1
    print(acc/len(scores))
    df.insert(loc=2, column='prediction', value=predictions)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    save_path = os.path.join(args.save_dir, 'result.csv')
    df.to_csv(save_path, index=False)



if __name__ == '__main__':
    args = parse_args()
    test(args)
