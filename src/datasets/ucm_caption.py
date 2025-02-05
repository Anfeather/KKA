from torch.utils.data import Subset,DataLoader
from PIL import Image
from torchvision.datasets import MNIST
from src.base.torchvision_dataset import TorchvisionDataset
from .preprocessing import create_semisupervised_setting
from .MSCOCO_split import UCM_caption_read

import torch
import torchvision.transforms as transforms
import random


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        # print("2", self.transform(x).shape)
        return [self.transform(x), self.transform(x)]

class UCM_Caption_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class: int = 0, known_outlier_class: int = 1, n_known_outlier_classes: int = 0,
                 ratio_known_normal: float = 0.0, ratio_known_outlier: float = 0.0, ratio_pollution: float = 0.0):
        super().__init__(root)

        # Define normal and outlier classes
        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = tuple([known_outlier_class])


        if n_known_outlier_classes == 0:
            self.known_outlier_classes = ()
        elif n_known_outlier_classes == 1:
            self.known_outlier_classes = tuple([known_outlier_class])
        else:
            self.known_outlier_classes = tuple(random.sample(self.outlier_classes, n_known_outlier_classes))


        # Set when training with data set 0
        user_dict = {0: [21,20,0],
                     1: [21, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}

        Anomalyfold = 0

        # Set when iterating over data set 0
        # user_dict = {0: [21, 20, 0],
        #              1: [21, 20, 0]}
        size = 32

        norm_layer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transform_test = [transforms.Resize((size, size)), transforms.ToTensor(), norm_layer]

        transform_train = [transforms.RandomResizedCrop(size, scale=(0.2, 1.0)), transforms.RandomHorizontalFlip(),
                           transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                           transforms.RandomGrayscale(p=0.2), transforms.ToTensor(), norm_layer]

        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(transform_test)

        train_set = UCM_caption_read(self.root ,'train', user_dict, 0, Anomalyfold,transform_train)

        idx, _, semi_targets = create_semisupervised_setting(train_set.targets.cpu().data.numpy(), self.normal_classes,
                                                             self.outlier_classes, self.known_outlier_classes,
                                                             ratio_known_normal, ratio_known_outlier, ratio_pollution)

        train_set.semi_targets[idx] = torch.tensor(semi_targets)  # set respective semi-supervised labels
        self.train_set = Subset(train_set, idx)

        # Get test set
        self.test_set = UCM_caption_read(self.root,  'test', user_dict, 1, Anomalyfold,transform_test)
        

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                  num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test,
                                 num_workers=num_workers, drop_last=False)
        return train_loader, test_loader




    def getdcaptions(self,indexes):
        return self.test_set.getcaptions(indexes)




