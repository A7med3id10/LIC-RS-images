import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class UCMercedDataset(Dataset):
    """
    Dataset loader for UC Merced Land Use dataset based on CSV splits.
    """
    def __init__(self, csv_path, root_dir, transform=None, sanity_check=True):
        self.data_frame = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform
        
        if sanity_check:
            self._run_sanity_check()

    def _run_sanity_check(self):
        print(f"Running sanity check on {len(self.data_frame)} images...")
        for idx in range(len(self.data_frame)):
            row = self.data_frame.iloc[idx]
            
            # path   
            img_rel_path = row['path'] if 'path' in self.data_frame.columns else row.iloc[0]
            img_path = os.path.join(self.root_dir, img_rel_path)
            
            try:
                with Image.open(img_path) as img:
                    img.verify()
            except Exception as e:
                raise RuntimeError(f"Corrupt or unreadable image found at {img_path}: {e}")
        print("Sanity check passed successfully! All images are valid.")

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        row = self.data_frame.iloc[idx]
        
        # path
        img_rel_path = row['path'] if 'path' in self.data_frame.columns else row.iloc[0]
        img_path = os.path.join(self.root_dir, img_rel_path)
        
        # Open RGB image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image

def get_dataloaders(data_root, csv_dir, batch_size=16):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = UCMercedDataset(
        csv_path=os.path.join(csv_dir, 'train.csv'),
        root_dir=data_root,
        transform=transform,
        sanity_check=False # False for fast training
    )
    
    val_dataset = UCMercedDataset(
        csv_path=os.path.join(csv_dir, 'val.csv'),
        root_dir=data_root,
        transform=transform,
        sanity_check=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    return train_loader, val_loader