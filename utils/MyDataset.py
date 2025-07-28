import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        #if torch.isnan(label):
            
        return self.features[idx], self.labels[idx]
