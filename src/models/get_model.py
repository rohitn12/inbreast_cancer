# import torch
# import torchvision
import torch.nn as nn
import torchvision.models as models
import segmentation_models_pytorch as smp
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

def initialize_classification_model(config):
    """
    intializes classification model based on the model name specified in the config 

    Args:
        config (_type_): config yaml from main

    Returns:
        _type_: pytorch model
    """
    model = config['classification']['model_name']
    n_classes = config['model']['n_classes_classification']
    if model == 'convnext':
        basemodel = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        basemodel.features[0][0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
        basemodel.classifier[2] = nn.Linear(in_features=768, out_features=3, bias=True) 


    elif model == 'efficient_net':
        basemodel = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        basemodel.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        basemodel.classifier[1] = nn.Linear(in_features=1280, out_features=3, bias=True)

    elif model == 'resnet34':
        basemodel = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        basemodel.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        basemodel.fc = nn.Linear(in_features=512, out_features=3, bias=True)
    else:
        print("Model not in list")
    return basemodel
    
def initialize_segmentation_models(config):
    """
    intializes segmentation model based on the model name specified in the config 

    Args:
        config (_type_): config yaml from main

    Returns:
        _type_: pytorch model
    """
    architecture = config['segmentation']['architecture']
    encoder = config['segmentation']['encoder']
    classification = config['segmentation']['classification']
    n_classes = config['model']['n_classes']
    in_channels = config['model']['in_channels']

    if classification:
        aux_params = dict(
                        pooling='avg',        # one of 'avg', 'max'
                        dropout=0.5,          # dropout ratio, default is None
                        activation='softmax', # activation function, default is None
                        classes=1,            # define number of output labels
                    )
    else:
        aux_params = None
    if architecture == "Unet":    
        basemodel = smp.Unet(encoder_name=encoder, in_channels=in_channels, classes=n_classes, aux_params=aux_params)
    elif architecture == 'Unet++':
        basemodel = smp.UnetPlusPlus(encoder_name=encoder, in_channels=in_channels, classes=n_classes, aux_params=aux_params)
    elif architecture == "FPN":
        basemodel = smp.FPN(encoder_name=encoder, in_channels=in_channels, classes=n_classes, aux_params=aux_params)
    elif architecture == 'PSPNet':
        basemodel = smp.PSPNet(encoder_name=encoder, in_channels=in_channels, classes=n_classes, aux_params=aux_params)
    elif architecture == 'segformer':
        basemodel = initialize_segformer(config)
    else:
        print("Model not in list")
    return basemodel


def load_model(config):
    """_summary_
    checks which model to initialize based on the config
    Args:
        config (_type_): config yaml from main

    Returns:
        _type_: pytorch model
    """
    if config['classification']['state'] == True:    
        model_name = config['classification']['model_name']
        model = initialize_classification_model(config)
    elif config['segmenetation']['state'] == True:
        model = initialize_segmentation_models(config)
    elif config['hybrid']['state'] == True:
        pass # to be implemented
    return model


def initialize_segformer(config):
    """
    intializes segformer model based on the model name specified in the config 

    Args:
        config (_type_): config yaml from main

    Returns:
        _type_: pytorch model
    """
    n_classes = config['model']['n_classes_seg']
    in_channels = config['model']['in_channels']
    l2i = {0:'clean', 1:'cancer'}
    i2l = {'clean':0 , 'cancer':1}
    model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5",
                                                            num_labels=2,num_channels=in_channels,
                                        label2id=l2i,id2label=i2l,semantic_loss_ignore_index=0,ignore_mismatched_sizes=True)
    return model
    