import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np
import cv2
import torch
from PIL import Image

class UCM_caption_read(Dataset):

    def __init__(self, data_folder, flag, user_dict, user_id,Anomalyfold,transform=None):
        data_distribution = user_dict[user_id]
        self.imgs = []
        self.captions = []
        self.labels = []
        self.paths = []
        self.flag = flag

        if flag == "train":
            # Fetch the data set in bit order and set the label
            for i in data_distribution:
                data_train_path = os.path.join(data_folder, str(i))
                with open(os.path.join(data_train_path,  "image" + '.json'), 'r') as j:
                    temp = json.load(j)
                    temp_len = len(temp)
                    self.imgs.extend([os.path.join(data_train_path, img[-15:]) for img in temp])
                # Load caption lengths (completely into memory)
                with open(os.path.join(data_train_path, "caption_sentence" + '.json'), 'r') as j:
                    self.captions.extend(json.load(j))
                # Normal samples are read first,0 represents normal samples, 1 represents anomaly samples, and folder 0 is anomaly samples
                if i == Anomalyfold:
                    self.labels.extend( [1] * temp_len )
                else:
                    self.labels.extend( [0] * temp_len )

        elif flag == "test":

            for i in data_distribution:
                if i ==Anomalyfold :
                    data_train_path = os.path.join(data_folder, str(i))
                    with open(os.path.join(data_train_path, "image" + '.json'), 'r') as j:
                        test_images = json.load(j)
                        test_images_len = len(test_images)
                        self.imgs.extend([os.path.join(data_train_path, img[-15:]) for img in test_images])
                    with open(os.path.join(data_train_path, "caption_sentence" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
                    self.labels.extend([1] * test_images_len)

                else:
                    data_train_path = os.path.join(data_folder, str(i))
                    with open(os.path.join(data_train_path,  "image_test"+ '.json'), 'r') as j:
                        test_images = json.load(j)
                        test_images_len = len(test_images)
                        self.imgs.extend([os.path.join(data_train_path, img[-15:]) for img in test_images])
                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "caption_sentence_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
                    if i in [20, 21]:
                        self.labels.extend([0] * test_images_len)
                    else:
                        self.labels.extend([1] * test_images_len)

        self.targets = torch.tensor(self.labels,dtype = torch.long)
        self.transform = transform
        # Total number of datapoints
        self.dataset_size = len(self.imgs)
        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = self.imgs[index]
        image_name = img[-9:-4]
        if self.transform is not None:
            img = cv2.imread(img)
            img = Image.fromarray(np.uint8(img))
            img = self.transform(img)
        # caption = self.captions[index*5]
        target = int(self.labels[index])
        semi_target = int(self.semi_targets[index])

        return img, target, semi_target, index

    def __len__(self):
        return self.dataset_size

    def getcaptions(self,indexes):
        All_caption = []
        for i in indexes:
            caption = self.captions[i*5:(i+1)*5]
            All_caption.append(caption)
        return All_caption

class Flower_detection(Dataset):

    def __init__(self, data_folder, flag, user_dict, user_id,Anomalyfold,transform=None):
        data_distribution = user_dict[user_id]
        self.imgs = []
        self.captions = []
        self.labels = []
        self.paths = []
        self.flag = flag

        if flag == "train":
            for i in data_distribution:
                data_train_path = os.path.join(data_folder, str(i))
                with open(os.path.join(data_train_path,  "image" + '.json'), 'r') as j:
                    temp = json.load(j)
                    temp_len = len(temp)
                    self.imgs.extend([os.path.join(data_train_path, img[-15:]) for img in temp])
                # Load caption lengths (completely into memory)
                with open(os.path.join(data_train_path, "text_caption" + '.json'), 'r') as j:
                    self.captions.extend(json.load(j))
                if i == Anomalyfold:
                    self.labels.extend( [1] * temp_len )
                else:
                    self.labels.extend( [0] * temp_len )

        elif flag == "test":

            for i in data_distribution:
                if i ==Anomalyfold :
                    data_train_path = os.path.join(data_folder, str(i))
                    with open(os.path.join(data_train_path, "image" + '.json'), 'r') as j:
                        test_images = json.load(j)
                        test_images_len = len(test_images)
                        self.imgs.extend([os.path.join(data_train_path, img[-15:]) for img in test_images])
                    with open(os.path.join(data_train_path, "text_caption" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
                    self.labels.extend([1] * test_images_len)

                else:
                    data_train_path = os.path.join(data_folder, str(i))
                    # data_train_path = data_folder +"/"+ str(i)
                    with open(os.path.join(data_train_path,  "image_test"+ '.json'), 'r') as j:
                        test_images = json.load(j)
                        test_images_len = len(test_images)
                        self.imgs.extend([os.path.join(data_train_path, img[-15:]) for img in test_images])
                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "text_caption_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
                    if i == 26:
                        self.labels.extend([0] * test_images_len)
                    else:
                        self.labels.extend([1] * test_images_len)


        self.targets = torch.tensor(self.labels,dtype = torch.long)
        self.transform = transform
        # Total number of datapoints
        self.dataset_size = len(self.imgs)
        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = self.imgs[index]
        image_name = img[-9:-4]
        if self.transform is not None:
            img = cv2.imread(img)
            img = Image.fromarray(np.uint8(img))
            img = self.transform(img)
        caption = self.captions[index*10]
        target = int(self.labels[index])
        semi_target = int(self.semi_targets[index])

        # return img, target, semi_target, index
        return img, target, semi_target, index


    def __len__(self):
        return self.dataset_size

    def getcaptions(self,indexes):
        All_caption = []
        for i in indexes:
            caption = self.captions[i*10:(i+1)*10]
            All_caption.append(caption)
        return All_caption

class mycifar_read(Dataset):

    def __init__(self, data_folder, flag, user_dict, user_id,Anomalyfold,transform=None):
        data_distribution = user_dict[user_id]
        self.imgs = []
        self.captions = []
        self.labels = []
        self.paths = []
        self.flag = flag

        if flag == "train":
            for i in data_distribution:
                data_train_path = os.path.join(data_folder, str(i))
                with open(os.path.join(data_train_path,  "image" + '.json'), 'r') as j:
                    temp = json.load(j)
                    temp_len = len(temp)
                    self.imgs.extend([os.path.join(data_train_path, img[-15:]) for img in temp])
                # Load caption lengths (completely into memory)
                with open(os.path.join(data_train_path, "caption_sentence" + '.json'), 'r') as j:
                    self.captions.extend(json.load(j))
                if i == Anomalyfold:
                    self.labels.extend( [1] * temp_len )
                else:
                    self.labels.extend( [0] * temp_len )

        elif flag == "test":

            for i in data_distribution:
                if i ==Anomalyfold :
                    data_train_path = os.path.join(data_folder, str(i))
                    with open(os.path.join(data_train_path, "image" + '.json'), 'r') as j:
                        test_images = json.load(j)
                        test_images_len = len(test_images)
                        self.imgs.extend([os.path.join(data_train_path, img[-15:]) for img in test_images])
                    with open(os.path.join(data_train_path, "caption_sentence" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
                    self.labels.extend([1] * test_images_len)
                else:
                    data_train_path = os.path.join(data_folder, str(i))
                    # data_train_path = data_folder +"/"+ str(i)
                    with open(os.path.join(data_train_path,  "image_test"+ '.json'), 'r') as j:
                        test_images = json.load(j)
                        test_images_len = len(test_images)
                        self.imgs.extend([os.path.join(data_train_path, img[-15:]) for img in test_images])
                    # Load caption lengths (completely into memory)
                    with open(os.path.join(data_train_path, "caption_sentence_test" + '.json'), 'r') as j:
                        self.captions.extend(json.load(j))
                    if i in [5, 20, 25, 84, 94]:
                        self.labels.extend([0] * test_images_len)
                    else:
                        self.labels.extend([1] * test_images_len)

        self.targets = torch.tensor(self.labels,dtype = torch.long)
        self.transform = transform
        # Total number of datapoints
        self.dataset_size = len(self.imgs)
        self.semi_targets = torch.zeros_like(self.targets)    
        
    def __getitem__(self, index):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = self.imgs[index]
        if self.transform is not None:
            img = cv2.imread(img)
            img = Image.fromarray(np.uint8(img))
            img = self.transform(img)
        target = int(self.labels[index])
        semi_target = int(self.semi_targets[index])

        return img, target, semi_target, index

    def __len__(self):
        return self.dataset_size

    def getcaptions(self,indexes):
        All_caption = []
        for i in indexes:
            caption = self.captions[i*1:(i+1)*1]
            All_caption.append(caption)
        return All_caption