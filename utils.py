
import os
import torch
import pandas as pd

def save_checkpoint(state, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def load_checkpoint(filename, device='cpu'):
    return torch.load(filename, map_location=device)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_logs(log_dict, filepath):
    df = pd.DataFrame(log_dict)
    df.to_csv(filepath, index=False)
