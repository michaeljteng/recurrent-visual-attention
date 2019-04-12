import numpy as np
import pickle
import os

from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
#from torchvision.datasets.utils import makedir_exist_ok


def normalize_attention_loc(integers, w=28, h=28, T=1):
    """
    Transforms tensor of image locations represented by the top left pixel
    number into coordinates of centre in [-1, 1]^2.
    """
    assert integers.max() < w*h

    integers = integers.type(torch.FloatTensor)

    # get pixel coords with top left as (0, 0) and bottom right as (x, y) = (w-1, h-1)
    pixel_x, pixel_y = integers % w, integers // w

    # normalise to between (0, 0) (inclusive) and (1, 1) (exclusive)
    pixel_x = pixel_x / w
    pixel_y = pixel_y / h

    # shift locations to centre of bins
    pixel_x = pixel_x + 1/(2*w)
    pixel_y = pixel_y + 1/(2*h)

    # move to between (-1, -1) and (1, 1) (both exclusive)
    pixel_x = 2*pixel_x - 1
    pixel_y = 2*pixel_y - 1

    return torch.cat([pixel_x.unsqueeze(-1), pixel_y.unsqueeze(-1)], dim=-1)



class AttentionTargetDataset(Dataset):
    data_file_name = 'training.pt'

    def __init__(self, root, transform=None, label_transform=None, attention_target_transform=None, loadfrom=None, max_dataset_size=100000000):
        self.processed_folder = os.path.join(root, "processed")
        self.training_path = os.path.join(self.processed_folder, self.data_file_name)
        self.transform = transform
        self.label_transform = label_transform
        self.attention_target_transform = attention_target_transform

        if loadfrom is not None:
            self._process(loadfrom, max_dataset_size)

        if not self._processed_data_exists():
            raise RuntimeError('No processed data found.')

        self.images, self.labels, self.attention_targets, self.posterior_targets =\
            torch.load(self.training_path)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        attention_target = self.attention_targets[index]
        posterior_target = self.posterior_targets[index]

        # image = Image.fromarray(image.numpy(), mode='L')
        image = image.view(1, 28, 28)

        if self.transform is not None:
            image = self.transform(image)

        if self.label_transform is not None:
            label = self.label_transform(label)

        if self.attention_target_transform is not None:
            attention_target = self.attention_target_transform(attention_target)

        return image, label, attention_target, posterior_target

    def _process(self, raw_path, max_dataset_size):
        """
        Loads data from specified folder, which should have format:
         - <raw_path>
           - images
             - 0.png
             - 1.png
             ...
           - logs
             - 0.p
             - 1.p
             ...
        where each *.p is a pickled dictionary containing optimal attention
        locations in format ... with key ...
        """
        os.makedirs(self.processed_folder, exist_ok=True)

        images = torch.Tensor([])
        labels = torch.LongTensor([])
        attention_targets = torch.Tensor([])
        posterior_targets = torch.Tensor([])

        for img_path, log_path in self._iter_raw_files(raw_path, max_dataset_size):
            print(f"loading {img_path}")
            image = torch.Tensor(np.array(Image.open(img_path))).view(1, 28, 28)
            log = pickle.load(open(log_path, 'rb'))
            label = log["true_digit"].view(1)
            attention_target = torch.Tensor(log["designs"]).view(1, -1)
            posterior_target = torch.cat([dp.view(1, 10) for dp in log['digit_posteriors']]).view(1, -1, 10)

            images = torch.cat((images, image), dim=0)
            labels = torch.cat((labels, label), dim=0)
            attention_targets = torch.cat((attention_targets, attention_target), dim=0)
            posterior_targets = torch.cat((posterior_targets, posterior_target), dim=0)

        data = (images, labels, attention_targets, posterior_targets)
        with open(self.training_path, 'wb') as f:
            torch.save(data, f)

    def _iter_raw_files(self, raw_path, maximum=0):
        i = 0
        while True:
            img_path = os.path.join(raw_path, 'images', f"{i}.png")
            log_path = os.path.join(raw_path, 'logs', f"{i}.p")
            if os.path.exists(img_path):
                yield img_path, log_path
                i += 1
                if i == maximum:
                    break
            else:
                break

    def _processed_data_exists(self):
        return os.path.exists(self.training_path)

    def __len__(self):
        return len(self.images)

def MixtureDataset(Dataset):
    def __init__(self, targets_dataset, other_dataset, targets_prob):
        self.targets_dataset = targets_dataset
        self.other_dataset = other_dataset
        self.mixture_prob = mixture_prob

        self.num_targets_due = 0

    def _use_target(self):
        self.num_targets_due += targets_prob
        if self.num_targets_due >= 1:
            self.num_targets_due -= 1
            return True
        return False

    def __getitem__(self, index):
        if self._use_target():
            return self.targets_dataset.get_item(index)
        return self.other_dataset.get_item(index)


if __name__ == '__main__':
    import argparse

    arg_lists = []
    parser = argparse.ArgumentParser(description='Attention target dataset')
    parser.add_argument('raw_data_folder', type=str,
                        help='path of raw data folder')
    parser.add_argument('-max_size', type=int, help='maximum number of data points to take')
    args = parser.parse_args()

    AttentionTargetDataset("attention_target_data", loadfrom=args.raw_data_folder, max_dataset_size=args.max_size)
