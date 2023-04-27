from PIL import Image
from torch.utils.data import Dataset
from skimage import io
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import numpy as np
import os
torch.manual_seed(0)
np.random.seed(0)


class Coffee_Beans(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        super(Coffee_Beans, self).__init__()
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        image = io.imread(img_path)/255.0
        y_label = torch.tensor(int(self.annotations.iloc[index, 0]))
        if self.transform:
            image = self.transform(image)
        return image, y_label

    def train_test_loader(self):
        train_ind, test_ind = train_test_split(list(range(len(self))), shuffle=True,
                                               test_size=0.2,
                                               stratify=self.annotations.loc[:, 'class index'].tolist())
        train_dataset = torch.utils.data.Subset(self, train_ind)
        test_dataset = torch.utils.data.Subset(self, test_ind)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
        return train_loader, test_loader