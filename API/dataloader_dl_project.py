import gzip
import random
import numpy as np
import torch
import torch.utils.data as data
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, is_train = True, n_frames_input=11, n_frames_output=11):
        self.image_names = sorted(os.listdir(folder_path))
        self.image_paths = [os.path.join(folder_path, name) for name in self.image_names]
        self.n_frames_input = n_frames_input
        self.n_frames_output = n_frames_output

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        folder_path = self.image_paths[index]
        image_filenames = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

        # Load each image as a PyTorch tensor and append it to a list
        images = []
        for filename in image_filenames:
            image_path = os.path.join(folder_path, filename)
            try:
                with Image.open(image_path) as image:
                    image = transforms.ToTensor()(image)
                    images.append(image)
            except:
                print("error occur")
                print(image_path)
                images.append(torch.zeros(3, 160, 240))

        # Stack the list of tensors into a single tensor with an additional batch dimension
        batch_of_images = torch.stack(images, dim=0)
        assert batch_of_images.shape[0] == self.n_frames_input + self.n_frames_output
        return batch_of_images[:self.n_frames_input], batch_of_images[-self.n_frames_output:]

def load_data(
        batch_size, val_batch_size,
        data_root, num_workers):

    train_set = ImageFolderDataset(data_root + '/unlabeled', is_train=True,
                            n_frames_input=11, n_frames_output=11)
    test_set = ImageFolderDataset(data_root + '/val', is_train=False,
                           n_frames_input=11, n_frames_output=11)

    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    dataloader_validation = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    mean, std = 0, 1
    return dataloader_train, dataloader_validation, dataloader_test, mean, std
