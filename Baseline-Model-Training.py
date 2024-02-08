#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
import matplotlib.pyplot as plt
import os
import argparse

from torchvision.datasets.folder import default_loader
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from torchsummary import summary


# In[ ]:

parser = argparse.ArgumentParser(description='Process command line arguments for your script.')

# Add the dataset name argument
parser.add_argument('-d', '--dataset', type=str, required=True,
                    help='Name of the dataset to use.')

# Add the output directory argument
parser.add_argument('-o', '--output_dir', type=str, required=False,
                    help='Output directory for the model.')

# Add the logs file argument
parser.add_argument('-l', '--logs_file', type=str, required=False,
                    help='Logs file for storing logs.')

# Parse the command line arguments
args = parser.parse_args()



# ## 1 - Dataloader for CUB Dataset

# In[3]:
# If want to train for the first dataset
if args.dataset == '1':
        
    print("Loading CUB Dataset: ")
    class CUBDataset(torchvision.datasets.ImageFolder):
        """
        Dataset class for CUB Dataset
        """

        def __init__(self, image_root_path, caption_root_path=None, split="train", *args, **kwargs):
            """
            Args:
                image_root_path:      path to dir containing images and lists folders
                caption_root_path:    path to dir containing captions
                split:          train / test
                *args:
                **kwargs:
            """
            image_info = self.get_file_content(f"{image_root_path}/images.txt")
            self.image_id_to_name = {y[0]: y[1] for y in [x.strip().split(" ") for x in image_info]}
            split_info = self.get_file_content(f"{image_root_path}/train_test_split.txt")
            self.split_info = {self.image_id_to_name[y[0]]: y[1] for y in [x.strip().split(" ") for x in split_info]}
            self.split = "1" if split == "train" else "0"
            self.caption_root_path = caption_root_path

            super(CUBDataset, self).__init__(root=f"{image_root_path}/images", is_valid_file=self.is_valid_file,
                                            *args, **kwargs)

        def is_valid_file(self, x):
            return self.split_info[(x[len(self.root) + 1:])] == self.split

        @staticmethod
        def get_file_content(file_path):
            with open(file_path) as fo:
                content = fo.readlines()
            return content


    # In[4]:


    data_root = "/apps/local/shared/CV703/datasets/CUB/CUB_200_2011"

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # write data transform here as per the requirement
    data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    train_dataset_cub = CUBDataset(image_root_path=f"{data_root}", transform=data_transform, split="train")
    test_dataset_cub = CUBDataset(image_root_path=f"{data_root}", transform=data_transform, split="test")


    # load in into the torch dataloader to get variable batch size, shuffle 
    train_loader_cub = torch.utils.data.DataLoader(train_dataset_cub, batch_size=32, drop_last=True, shuffle=True)
    test_loader_cub = torch.utils.data.DataLoader(test_dataset_cub, batch_size=32, drop_last=False, shuffle=False)


    # ### Test the dataloader

    # In[5]:


    len(train_dataset_cub), len(test_dataset_cub)


    # In[6]:


    len(train_loader_cub), len(test_loader_cub)


    # In[6]:


    for i, (inputs, labels) in enumerate(train_loader_cub):
        print(inputs.shape)
        print(labels)
        print('='*50)
        break



# ## 2 - Dataloader for FGVC Aircraft Dataset

elif args.dataset == '2':
    print("Loading FGVC + CUB Dataset: ")
    
    print("First Loading CUB Dataset: ")
    class CUBDataset(torchvision.datasets.ImageFolder):
        """
        Dataset class for CUB Dataset
        """

        def __init__(self, image_root_path, caption_root_path=None, split="train", *args, **kwargs):
            """
            Args:
                image_root_path:      path to dir containing images and lists folders
                caption_root_path:    path to dir containing captions
                split:          train / test
                *args:
                **kwargs:
            """
            image_info = self.get_file_content(f"{image_root_path}/images.txt")
            self.image_id_to_name = {y[0]: y[1] for y in [x.strip().split(" ") for x in image_info]}
            split_info = self.get_file_content(f"{image_root_path}/train_test_split.txt")
            self.split_info = {self.image_id_to_name[y[0]]: y[1] for y in [x.strip().split(" ") for x in split_info]}
            self.split = "1" if split == "train" else "0"
            self.caption_root_path = caption_root_path

            super(CUBDataset, self).__init__(root=f"{image_root_path}/images", is_valid_file=self.is_valid_file,
                                            *args, **kwargs)

        def is_valid_file(self, x):
            return self.split_info[(x[len(self.root) + 1:])] == self.split

        @staticmethod
        def get_file_content(file_path):
            with open(file_path) as fo:
                content = fo.readlines()
            return content


    # In[4]:


    data_root = "/apps/local/shared/CV703/datasets/CUB/CUB_200_2011"

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # write data transform here as per the requirement
    data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    train_dataset_cub = CUBDataset(image_root_path=f"{data_root}", transform=data_transform, split="train")
    test_dataset_cub = CUBDataset(image_root_path=f"{data_root}", transform=data_transform, split="test")


    # load in into the torch dataloader to get variable batch size, shuffle 
    train_loader_cub = torch.utils.data.DataLoader(train_dataset_cub, batch_size=32, drop_last=True, shuffle=True)
    test_loader_cub = torch.utils.data.DataLoader(test_dataset_cub, batch_size=32, drop_last=False, shuffle=False)


    class FGVCAircraft(VisionDataset):
        """
        FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
                creates from test set.
            class_type (string, optional): choose from ('variant', 'family', 'manufacturer').
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
        """
        
        class_types = ('variant', 'family', 'manufacturer')
        splits = ('train', 'val', 'trainval', 'test')
        img_folder = os.path.join('data', 'images')

        def __init__(self, root, train=True, class_type='variant', transform=None,
                    target_transform=None):
            super(FGVCAircraft, self).__init__(root, transform=transform, target_transform=target_transform)
            split = 'trainval' if train else 'test'
            if split not in self.splits:
                raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                    split, ', '.join(self.splits),
                ))
            if class_type not in self.class_types:
                raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                    class_type, ', '.join(self.class_types),
                ))

            self.class_type = class_type
            self.split = split
            self.classes_file = os.path.join(self.root, 'data',
                                            'images_%s_%s.txt' % (self.class_type, self.split))

            (image_ids, targets, classes, class_to_idx) = self.find_classes()
            samples = self.make_dataset(image_ids, targets)

            self.loader = default_loader

            self.samples = samples
            self.classes = classes
            self.class_to_idx = class_to_idx

        def __getitem__(self, index):
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target

        def __len__(self):
            return len(self.samples)

        def find_classes(self):
            # read classes file, separating out image IDs and class names
            image_ids = []
            targets = []
            with open(self.classes_file, 'r') as f:
                for line in f:
                    split_line = line.split(' ')
                    image_ids.append(split_line[0])
                    targets.append(' '.join(split_line[1:]))

            # index class names
            classes = np.unique(targets)
            class_to_idx = {classes[i]: i for i in range(len(classes))}
            targets = [class_to_idx[c] for c in targets]
            
            # Modify class index as we are going to concat to CUB dataset
            num_cub_classes = len(train_dataset_cub.class_to_idx)
            targets = [t + num_cub_classes for t in targets]

            return image_ids, targets, classes, class_to_idx

        def make_dataset(self, image_ids, targets):
            assert (len(image_ids) == len(targets))
            images = []
            for i in range(len(image_ids)):
                item = (os.path.join(self.root, self.img_folder,
                                    '%s.jpg' % image_ids[i]), targets[i])
                images.append(item)
            return images


    # In[8]:


    data_root = "/apps/local/shared/CV703/datasets/fgvc-aircraft-2013b"

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # write data transform here as per the requirement
    data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    train_dataset_aircraft = FGVCAircraft(root=f"{data_root}", transform=data_transform, train=True)
    test_dataset_aircraft = FGVCAircraft(root=f"{data_root}", transform=data_transform, train=False)


    # load in into the torch dataloader to get variable batch size, shuffle 
    train_loader_aircraft = torch.utils.data.DataLoader(train_dataset_aircraft, batch_size=32, drop_last=True, shuffle=True)
    test_loader_aircraft = torch.utils.data.DataLoader(test_dataset_aircraft, batch_size=32, drop_last=False, shuffle=False)

    for i, (inputs, labels) in enumerate(train_loader_aircraft):
        print(inputs.shape)
        print(labels)
        print('='*50)
        break


    # ## Concatenate CUB Birds and FGVC Aircraft Datasets

    # In[12]:


    from torch.utils.data import ConcatDataset 


    # In[13]:


    concat_dataset_train = ConcatDataset([train_dataset_cub, train_dataset_aircraft])
    concat_dataset_test = ConcatDataset([test_dataset_cub, test_dataset_aircraft])

    concat_loader_train = torch.utils.data.DataLoader(
                concat_dataset_train,
                batch_size=32, shuffle=True,
                num_workers=1, pin_memory=True
                )
    concat_loader_test = torch.utils.data.DataLoader(
                concat_dataset_test,
                batch_size=32, shuffle=False,
                num_workers=1, pin_memory=True
                )

    print("Size of Data: " , len(concat_dataset_train), len(concat_dataset_test))

# If want to train for the third dataset
elif args.dataset == '3':
    print("Loading Food Dataset: ")
    # ## 3. Dataloader for Food Dataset

    # In[19]:
    import os
    import torch
    import torchvision
    from torchvision import datasets, models, transforms
    from PIL import Image
    import pandas as pd


    # In[29]:


    data_dir = "/apps/local/shared/CV703/datasets/FoodX/food_dataset"

    split = 'train'
    train_df = pd.read_csv(f'{data_dir}/annot/{split}_info.csv', names= ['image_name','label'])
    train_df['path'] = train_df['image_name'].map(lambda x: os.path.join(f'{data_dir}/{split}_set/', x))


    split = 'val'
    val_df = pd.read_csv(f'{data_dir}/annot/{split}_info.csv', names= ['image_name','label'])
    val_df['path'] = val_df['image_name'].map(lambda x: os.path.join(f'{data_dir}/{split}_set/', x))


    # In[30]:


    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224))
            
        ])


    # In[31]:


    class FOODDataset(torch.utils.data.Dataset):
        def __init__(self, dataframe):
            self.dataframe = dataframe

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, index):
            row = self.dataframe.iloc[index]
            return (
                data_transform(Image.open(row["path"])), row['label']
            )


    # In[32]:


    train_dataset = FOODDataset(train_df)
    val_dataset = FOODDataset(val_df)

    # load in into the torch dataloader to get variable batch size, shuffle 
    train_loader_food = torch.utils.data.DataLoader(train_dataset, batch_size=32, drop_last=True, shuffle=True)
    val_loader_food = torch.utils.data.DataLoader(val_dataset, batch_size=32, drop_last=False, shuffle=True)


    # In[33]:


    print(len(train_dataset), len(val_dataset))


    # In[25]:


    print(len(train_loader_food), len(val_loader_food))


    # In[26]:


    for i, (inputs, labels) in enumerate(val_loader_food):
        print(inputs.shape)
        print(labels)
        print('='*50)
        break

else:
    print("Please pass correct argument.")

# In[ ]:





# ## Baseline Model

# In[5]:


from torch import nn
from torch import Tensor
from typing import List
from torchvision.ops import StochasticDepth
from tqdm import tqdm

# In[9]:


import torch
from torchsummary import summary

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
device = get_default_device()

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
                
early_stopping = EarlyStopping(tolerance=5, min_delta=0.010)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

  
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


# In[10]:
# Loading corresponding dataset against the provided input
if args.dataset == '1':    
    train_loader = DeviceDataLoader(train_loader_cub, device)
    val_loader = DeviceDataLoader(test_loader_cub, device)
    
elif args.dataset == '2':    
    train_loader = DeviceDataLoader(concat_loader_train, device)
    val_loader = DeviceDataLoader(concat_loader_test, device)
    
elif args.dataset == '3':    
    train_loader = DeviceDataLoader(train_loader_food, device)
    val_loader = DeviceDataLoader(val_loader_food, device)
    
else:   
    print("Specify correct dataset.")


# In[16]:

import torch
from torchvision.models import convnext_large


# Initialize baseline model
def initialize_convnext_large(num_classes): 
    model = convnext_large(pretrained=True) 
    num_features = model.classifier[2].in_features 
    model.classifier[2] = nn.Linear(num_features, num_classes) 
    return model


# Function to calculate accuracy
def accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    acc = correct / total
    return acc

# Function to train the model
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Use tqdm to create a progress bar
    with tqdm(total=len(train_loader), desc="Training", unit="batch") as pbar:
        for inputs, labels in train_loader:
            inputs, labels = to_device(inputs, device), to_device(labels, device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (accuracy(outputs, labels) * labels.size(0))
            total_samples += labels.size(0)

            pbar.update(1)  # Update the progress bar

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc

# Function to validate the model
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Use tqdm to create a progress bar
    with tqdm(total=len(val_loader), desc="Validation", unit="batch") as pbar:
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = to_device(inputs, device), to_device(labels, device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)
                pbar.update(1)  # Update the progress bar

    avg_loss = total_loss / len(val_loader)
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy


# Initialize ConvNext Large model according to each dataset
if args.dataset == '1':
    model = initialize_convnext_large(200)
    
elif args.dataset == '2':
    model = initialize_convnext_large(300)
    
elif args.dataset == '3':
    model = initialize_convnext_large(251)
    print(model)
    
else:
    print("Specify correct number of classes...!")

# Move the model to the device
model = to_device(model, device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
# Define scheduler
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# In[45]:


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, train_loss, validation_loss):
        if (validation_loss - train_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
                
early_stopping = EarlyStopping(tolerance=5, min_delta=0.1)


import datetime

# Train the model with tqdm for progress bar
num_epochs = 100
correct = 0
total = 0
best_accuracy=0

# Final code for training and testing
log_file_path = f"./logs-model-{args.dataset}.txt"
with open(log_file_path, "w") as log_file:
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        avg_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {round(avg_loss , 4)}, Training Accuracy: {round(train_accuracy , 4)}%")

        # Evaluation
        model.eval()
        correct_test = 0
        total_test = 0
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Testing"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        test_accuracy = 100 * correct_test / total_test
        print(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {round(avg_val_loss , 4)}, Test Accuracy: {round(test_accuracy , 4)}%")

        # Update best accuracy and save model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            save_path = os.path.join('./Models', f"Model-{args.dataset}.pth")
            torch.save(model, save_path)

        # early stopping
        early_stopping(avg_loss, avg_val_loss)
        if early_stopping.early_stop:
            print("We are at epoch:", i)
            break
        # Write to log file
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{current_time} - Epoch {epoch + 1}: Training Loss: {round(avg_loss , 4)}, Training Accuracy = {round(train_accuracy , 4)}%, Test Loss: {round(avg_val_loss , 4)}, Test Accuracy = {round(test_accuracy , 4)}%\n")

        scheduler.step()
print(f"Our Best Test Accuracy Is : {best_accuracy}%")

