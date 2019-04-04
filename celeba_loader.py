from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
import torch
import os
import random
import numpy as np


class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
        else:
            self.num_images = len(self.test_dataset)

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = []
            for attr_name in self.selected_attrs:
                idx = self.attr2idx[attr_name]
                label.append(values[idx] == '1')

            if (i+1) < 2000:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print('Finished preprocessing the CelebA dataset...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        return self.transform(image), torch.LongTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_train_celeba_loader(image_dir, 
                            attr_path, 
                            selected_attrs, 
                            crop_size=178, 
                            image_size=128, 
                            batch_size=16, 
                            dataset='CelebA', 
                            mode='train', 
                            num_workers=1, 
                            valid_size=0.1,
                            ):
    """Build and return a data loader."""
    transform = []
    transform.append(T.Grayscale(num_output_channels=1))
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    print("image size being used is: ", image_size)
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)

    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    num_total = len(dataset)
    indices = list(range(num_total))
    test_split = int(np.floor(valid_size*2 * num_total))
    remainder_indices, test_idx = indices[test_split:], indices[:test_split]

    num_train = len(remainder_indices)
    valid_split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = remainder_indices[valid_split:], remainder_indices[:valid_split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  sampler=train_sampler,  
                                  num_workers=num_workers)

    valid_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  sampler=valid_sampler,  
                                  num_workers=num_workers)

    test_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  sampler=test_sampler,  
                                  num_workers=num_workers)
    return (train_loader, valid_loader, test_loader)
