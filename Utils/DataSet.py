import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
# This file creates the custom data set taking the Dataset
# by overriding the PyTorch abstract class DataSet.
# Data Set Format used : { Image : Double Tensor, Label : Int Tensor }

class SurfaceDataSet(Dataset):
    """Steel Surface Defect Classification Dataset NEU-CLS-64"""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file containing images and labels
        """
        self.dataSet = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.dataSet)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.dataSet.iloc[idx, 0])
        img =Image.open(img_name)
        image = np.zeros((1,img.size[0],img.size[1]))
        image[0,:,:] = np.array(img)
        label = int(self.dataSet.iloc[idx, 1:])
        sample = {'image': image, 'label': label}
        return sample