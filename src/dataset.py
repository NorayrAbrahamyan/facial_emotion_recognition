import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class FER2013Dataset(Dataset):
  def __init__(self, csv_file, transform=None):
    self.data = pd.read_csv(csv_file)
    self.transform = transform

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    label = int(self.data.iloc[idx, 0])
    pixel_string = self.data.iloc[idx, 1].strip()
    image = np.fromstring(pixel_string, sep=' ', dtype=np.uint8)
    if image.shape[0] != 2304:
        image = np.zeros(2304, dtype=np.uint8)
        
    image = image.reshape(48, 48)

    if self.transform:
      image = self.transform(image)

    return image, label