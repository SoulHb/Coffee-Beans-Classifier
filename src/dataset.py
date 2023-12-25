import pandas as pd
import torch
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
torch.manual_seed(0)
np.random.seed(0)


class CoffeeBeans(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
               Initializes a CoffeeBeans dataset.

               Args:
                   root_dir (str): The root directory path where the images are located.
                   csv_file (str): The path to the CSV file containing image annotations.
                   transform (callable, optional): A function/transform to be applied on the image. Default is None.
               """
        super(CoffeeBeans, self).__init__()
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """
                Returns the number of samples in the dataset.

                Returns:
                    int: The number of samples in the dataset.
                """
        return len(self.annotations)

    def __getitem__(self, index):
        """
                Retrieves and returns a sample from the dataset at the specified index.

                Args:
                    index (int): The index of the sample to retrieve.

                Returns:
                    tuple: A tuple containing the image and its corresponding label.
                """
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        # Load image using PIL
        image = Image.open(img_path).convert("RGB")

        # Normalize the image
        image = np.array(image) / 255.0
        # Convert to PIL Image
        image = Image.fromarray((image * 255).astype(np.uint8))
        y_label = torch.tensor(int(self.annotations.iloc[index, 0]))
        if self.transform:
            image = self.transform(image)
        return image, y_label

    def train_test_loader(self):
        """
                Splits the dataset into training and testing subsets and returns corresponding DataLoader instances.

                Returns:
                    tuple: A tuple containing DataLoader instances for the training and testing subsets.
                """
        train_ind, test_ind = train_test_split(list(range(len(self))), shuffle=True,
                                               test_size=0.2,
                                               stratify=self.annotations.loc[:, 'class index'].tolist())
        train_dataset = torch.utils.data.Subset(self, train_ind)
        test_dataset = torch.utils.data.Subset(self, test_ind)
        return train_dataset, test_dataset