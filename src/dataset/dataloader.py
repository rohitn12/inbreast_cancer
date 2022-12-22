import os
import cv2
import glob
import torch
import random 
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as TF

class INBreastDataloader(Dataset):
    """_summary_

    Args:
        Dataset (_type_): ImBreast dataset loader
    """
    def __init__(self, path_to_root, config,transforms=None, train=True):
        """_summary_

        Args:
            path_to_root (_type_): path to root data containing images in 'ALLDicoms' labels in 'masks'
            config (_type_): config yaml
            transforms (_type_, optional): image transformations. Defaults to None.
            train (bool, optional): train or test. Defaults to True.
        """
        random.seed(3)
        self.path_to_root = path_to_root
        self.path_to_images = path_to_root + 'AllDICOMs/'
        self.path_to_labels = path_to_root + 'masks/'
        self.list_of_image_paths = glob.glob(self.path_to_images + '*.dcm')
        path_to_xls = path_to_root + 'INbreast.xls'
        self.df = self.process_xls(path_to_xls)

        # preprocess df for compesating the label in imbalance in the classification
        if config['classification']['state']:
            self.list_of_image_paths = self.process_df_classification(self.list_of_image_paths, self.df)
        self.train = train
        self.transforms = transforms
    
    def __len__(self):
        return len(self.list_of_image_paths)

    def __getitem__(self, idx):
        # load image
        fname = self.list_of_image_paths[idx].split('/')[-1].split('_')[0]
        image = pydicom.dcmread(self.list_of_image_paths[idx])
        image = image.pixel_array
        image = (image / 4095).astype('float') # normalizing with 4095 since its the hhighest pixel value in dataset
        
        # load mask
        label_path = self.path_to_labels + fname + '.png'
        label = cv2.imread(label_path, 0) / 255.
        label = label.astype('float')
        # get classification label
        masked_df = self.df[self.df['image_id'].isin([fname])]
        birads_label = masked_df['birads']
        #view_label = masked_df['View']
        classification_label = np.array(birads_label)
        if transforms:
            image, label = self.get_transforms(image, label)
        # torch.Tensor(classification_label).type(torch.FloatTensor)
        return image.type(torch.FloatTensor), label.type(torch.FloatTensor),torch.Tensor(classification_label)

    def process_xls(self, path_to_xls):
        """Function loads xls files into pandas dataframe and processes the birads labels 

        Args:
            path_to_xls (_type_): path to labels xls files

        Returns:
            _type_: processed dataframe
        """
        df = pd.read_excel(
            path_to_xls, 
            dtype={'File Name': str},
            usecols=["File Name", "Bi-Rads", "View", "Laterality"])[:-2]

        df = df.rename(columns={"File Name": "image_id" , "Bi-Rads":"birads"})
        df = df[['image_id' , "birads", "View", "Laterality"]]
        df["birads"] = df["birads"].apply(lambda x : int(x) if str(x).isdigit() else int(str(x[0])))
        df["View"] = df["View"].apply(lambda x : 0 if x == "CC" else 1)       
        df["birads"] = df["birads"].apply(lambda x : 0  if x < 2 else x)
        df["birads"] = df["birads"].apply(lambda x : 1  if x == 2 or x == 3 else x)
        df["birads"] = df["birads"].apply(lambda x : 2  if x == 4 or x == 5 or x == 6 else x)
        return df

    def get_image_crop(self, image, label):
        """_summary_
        Args:
            image (_type_): image numpy array
            label (_type_): label numpy array

        Returns:
            _type_: cropped 640 x 240 images removing the background
        """

        _,thresh = cv2.threshold(image, 0.1, 1, cv2.THRESH_BINARY)
        indices = np.nonzero(thresh)

        # Find the minimum and maximum indices
        min_index = tuple(map(min, indices))
        max_index = tuple(map(max, indices))

        y_min = round(max(min_index[0] - 50, 0))
        x_min = round(max(min_index[1] - 50, 0))
        y_max = round(min(max_index[0] + 50, image.shape[0]))
        x_max = round(min(max_index[1] + 50, image.shape[1]))
        image = image[y_min: y_max, x_min: x_max]
        label = label[y_min: y_max, x_min: x_max]
        image = cv2.resize(image, (256, 640))
        label = cv2.resize(label, (256, 640))
        return image, label

    def get_transforms(self, image, label):
        """_summary_

        Args:
            image (_type_): image numpy array
            label (_type_): label numpy array

        Returns:
            _type_: returns torch tensor of transforms image and label
        """
        image , label = self.get_image_crop(image, label)
        
        image = TF.to_tensor(image)
        label = TF.to_tensor(label)

        apply_transform = random.random()
        if apply_transform > 0.4 or self.train == False:
            return image, label

        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)
        
        if random.random() > 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)

        if random.random() > 0.5:
            angle = random.randint(0,25)
            image = TF.rotate(image, angle)
            label = TF.rotate(label, angle)


        if random.random() > 0.5:
            image = TF.gaussian_blur(image, kernel_size=(3,3))
            label = TF.gaussian_blur(label, kernel_size=(3,3))

        return image, label

    def process_df_classification(self,list_of_images, df):
        """_summary_

        Args:
            list_of_images (_type_): list of all dicoms image
            df (_type_): xls pandas dataframe

        Returns:
            _type_: returns the updated dataframe to handle class imbalance 
        """
        processed_list = []
        count = 0
        for i in range(len(list_of_images)):
            fname = list_of_images[i].split('/')[-1].split('_')[0]
            if (df.loc[df['image_id'] == fname]['birads']  == 1).all():
                if count < 100:    
                    processed_list.append(list_of_images[i])
                    count += 1
            else:
                processed_list.append(list_of_images[i])
        return processed_list