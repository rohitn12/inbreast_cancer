import torch
import torch.nn as nn 


class SegAuxClassifier(nn.Module):
    """Auxilary network for segformer for classifying the type of detection

    Args:
        nn (_type_): _description_
    """
    
    def __init__(self):
        super(SegAuxClassifier,self).__init__()
        self.conv_layer  = nn.Sequential(
                        nn.Conv2d(in_channels=512 , out_channels=128, kernel_size=(2,2)),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Conv2d(in_channels=128 , out_channels=4, kernel_size=(2,2)),
                        nn.ReLU(),
                        nn.Dropout(p=0.5))
        self.fc_layer = nn.Sequential(
                        nn.Linear(in_features=432, out_features=128),
                        nn.ReLU(),
                        nn.Dropout(p=0.5),
                        nn.Linear(in_features=128, out_features=3))

    def forward(self,x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1) 
        x = self.fc_layer(x)
        return x