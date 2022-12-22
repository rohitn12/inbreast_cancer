import torch
import torch.nn as nn
import torch.nn.functional as F

#https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Combo-Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha = 0.8,gamma = 0.2,smooth = 1, weight=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1
    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss

#https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Combo-Loss
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7,smooth = 1,weight=None, size_average=True):
        super(TverskyLoss, self).__init__()
        self.alpha = 0.7
        self.beta = 1 - alpha
        self.smooth = smooth

    def forward(self, inputs, targets):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)  
        
        return 1 - Tversky

#https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook#Combo-Loss
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

def get_loss(config):
    """defines loss function for the model

    Args:
        config (_type_): yaml file 

    Returns:
        _type_: loss function
    """
    loss = config
    if loss == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss == "ce":
        return nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
    elif loss == 'dice':
        return TverskyLoss()
    elif loss == 'focal':
        return FocalLoss()
    elif loss == 'trescsy':
        return DiceLoss()
    else:
        print('Not supported')

