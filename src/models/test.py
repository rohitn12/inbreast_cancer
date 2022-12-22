import os
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from models.segformer_aux import SegAuxClassifier
from utils.loss import get_loss

def eval(model, testloader, config):
    """evalute trained model based on the config

    Args:
        model (_type_): pytorch model to evaluate
        testloader (_type_): pytorch dataloader for evaluating
        config (_type_): config yaml from main
    """
    #load models
    if config['classification']['state'] == True:
        eval_classification(model, testloader,config)
    elif config['segmentation']['model_name'] == 'segformer':
        eval_segformer(model, testloader,config)
    else:
        eval_segmentation(model, testloader, config)

def eval_classification(model, testloader,config):
    """evalute trained model to evaluate classification

    Args:
        model (_type_): pytorch model to evaluate
        testloader (_type_): pytorch dataloader for evaluating
        config (_type_): config yaml from main
    """
    logger = config['save_path'] + config['model_name'] + '/log_test.txt'
    save_path = config['save_path'] + config['model_name']
    best_accuracy = float('inf')
    device = config['device']
    
    total = 0
    correct = 0
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(testloader):
            image, mask, label = data
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        
    accuracy = correct / total
    progress = 'accuracy = {}'.format(accuracy)
    with open(logger , 'a') as f:
        f.write(progress + '\n')


def eval_segformer(model, testloader, config):
    """evalute trained model to evaluate segformer segmentation

    Args:
        model (_type_): pytorch model to evaluate
        testloader (_type_): pytorch dataloader for evaluating
        config (_type_): config yaml from main
    """
    logger = config['save_path'] + config['model_name'] + '/log_test.txt'
    save_path = config['save_path'] + config['model_name']
    save_dir = save_path + '/pred/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    segmentation_classfication = config['segmentation']['classification']
    device = config['device']
    total = 0
    correct = 0
    accuracy = 0
    if segmentation_classfication:
        aux = SegAuxClassifier().to(device)
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(testloader):
            image, mask, label = data
            image = image.to(device)
            label = label.to(device)
            mask = mask.to(device)
            o = model(pixel_values=image, labels=mask)
            logits = o['logits']
            logits = logits.detach().cpu()
            upsampled_logits = nn.functional.interpolate(
                logits,
                size=(640, 256), # (height, width)
                mode='bilinear',
                align_corners=False
            )

            for idx in range(len(upsampled_logits)):
                output_image = upsampled_logits[idx]
                pred_seg = torch.sigmoid(output_image[1])
                pred_seg_np = pred_seg.detach().numpy()
                pred_seg_np = np.array([(pred_seg_np * 255).astype(np.uint8)])
                #fname = get_file(testloader)
                plt.imsave('{}.png'.format(idx), pred_seg_np)

            if segmentation_classfication:
                encoder_output = model.segformer.encoder(image)  
                output_label = aux(encoder_output[0])                  
                _, predicted = torch.max(output_label.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

    accuracy = correct / total
    progress = 'accuracy = {}'.format(accuracy)
    with open(logger, 'a') as f:
        f.write(progress + '\n')
    

def eval_segmentation(model, testloader,config):
    """evalute trained model to evaluate segmentation

    Args:
        model (_type_): pytorch model to evaluate
        testloader (_type_): pytorch dataloader for evaluating
        config (_type_): config yaml from main
    """
    epochs = config['model']['epoch']
    logger = config['save_path'] + config['model_name'] + '/log.txt'
    save_path = config['save_path'] + config['model_name']

    save_dir = save_path + '/pred/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    segmentation_classfication = config['segmentation']['classification']
    device = config['device']
    total = 0
    correct = 0
    accuracy = 0
    with torch.no_grad():
        model.eval()
        for i, data in enumerate(testloader):
            image, mask, label = data
            image = image.to(device)
            label = label.to(device)
            mask = mask.to(device)

            if segmentation_classfication:                    
                output, output_label = model(image)
                _, predicted = torch.max(output_label.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            else:
                output = model(image)
            for idx in range(len(output)):
                output_image = output[idx]
                pred_seg = torch.sigmoid(output_image[1])
                pred_seg_np = pred_seg.detach().numpy()
                pred_seg_np = np.array([(pred_seg_np * 255).astype(np.uint8)])
                #fname = get_file(testloader)
                plt.imsave('{}.png'.format(idx), pred_seg_np)

    if total != 0:
        accuracy = correct / total           

    progress = ' accuracy = {}'.format()

    with open(logger , 'a') as f:
        f.write(progress + '\n')
