import torch
import yaml
import plistlib
import numpy as np
from skimage.draw import polygon
from dataset.dataloader import INBreastDataloader
from torch.utils.data import random_split, DataLoader

# taken from https://gist.github.com/jendelel/3a8e768a8eb9345d49f2a82d02946122?permalink_comment_id=3658239
def load_point(point_string):

    x, y = tuple([float(num) for num in point_string.strip('()').split(',')])
    return y, x


#taken from https://gist.github.com/jendelel/3a8e768a8eb9345d49f2a82d02946122?permalink_comment_id=3658239
def load_inbreast_mask(mask_path, imshape=(4084, 3328)):
    """
    This function loads a osirix xml region as a binary numpy array for 
    INBREAST dataset
    @mask_path : Path to the xml file
    @imshape : The shape of the image as an array e.g. [4084, 3328]
    return: numpy array where positions in the roi are assigned a value of 1.
    """
    mask = np.zeros(imshape)
    with open(mask_path, 'rb') as mask_file:
        plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)['Images'][0]
        numRois = plist_dict['NumberOfROIs']
        rois = plist_dict['ROIs']
        assert len(rois) == numRois
        for roi in rois:
            numPoints = roi['NumberOfPoints']
            points = roi['Point_px']
            assert numPoints == len(points)
            points = [load_point(point) for point in points]
            if len(points) <= 2:
                for point in points:
                    mask[int(point[0]), int(point[1])] = 1
            else:
                x, y = zip(*points)
                x, y = np.array(x), np.array(y)
                poly_x, poly_y = polygon(x, y, shape=imshape)
                mask[poly_x, poly_y] = 1
    return mask

def get_dataloaders(config, split_size=0.8):

    """
    Args:
        config (_type_): config yaml file
        split_size (float, optional): .train test split for the dataset Defaults to 0.8.

    Returns:
        _type_: _description_
    """
    
    if config['model']['train']:
        train_bool = True
    else:
        train_bool = False
    dataset = INBreastDataloader(config['path'], config, transforms=True, train=train_bool)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(5))
    train_dataloader = DataLoader(dataset, batch_size=config['model']['batch_size'], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config['model']['batch_size'], shuffle=False)
    return train_dataloader, test_dataloader



def load_yaml(path_to_config):
    """_summary_

    Args:
        path_to_config (_type_): path to yaml config file

    Returns:
        _type_: config file containing the paramters to the model
    """
    config = yaml.safe_load(open(path_to_config, 'r'))
    device_str = config.get('device', None)
    if device_str is not None:
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            device_str = 'cpu'
    else:
        device_str = "cuda" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    config['device'] = device
    return config
